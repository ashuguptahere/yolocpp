#pragma once
//
// YOLO13 — Lei et al., "YOLO13: Real-Time Object Detection with
// Hypergraph-Enhanced Adaptive Visual Perception" (mid-2025).
// Reference fork: github.com/iMoonLab/yolov13.
//
// Architectural diff vs v12:
//   - DSConv replaces strided Conv at L5/L7.
//   - DSC3k2 replaces C3k2 (depthwise-separable Bottlenecks; default k1=3, k2=7).
//   - HyperACE at L9 fuses the three backbone scales {L4, L6, L8} into a
//     hypergraph-attended map, then drives FullPAD_Tunnels (gated residuals)
//     into 5 distribution points across the neck.
//   - The neck has 33 layers (vs 22 for v8/v11/v12).
//   - Detect head is the standard v8 anchor-free DFL Detect with legacy=false
//     (v11-style nested DWConv cv3) on (P3, P4, P5) channels (256, 512, 1024)
//     scaled by `width`.
//
// 33-layer module list (yolov13.yaml):
//   00 Conv stem k=3 s=2                          (3      → 64)
//   01 Conv     k=3 s=2 g=2                       (64     → 128)
//   02 DSC3k2   n=2 dsc3k=False e=0.25            (128    → 256)
//   03 Conv     k=3 s=2 g=4                       (256    → 256)
//   04 DSC3k2   n=2 dsc3k=False e=0.25            (256    → 512)
//   05 DSConv   k=3 s=2                           (512    → 512)
//   06 A2C2f    n=4 a2=True area=4                (512    → 512)
//   07 DSConv   k=3 s=2                           (512    → 1024)
//   08 A2C2f    n=4 a2=True area=1                (1024   → 1024)
//   09 HyperACE([4,6,8]) n=2 num_hyperedges=8 dsc3k=True channel_adjust=True
//                                                  (out:   512)
//   10 Upsample(scale=2)                          (in/out 512)
//   11 DownsampleConv(in=512, channel_adjust=True) (512  → 1024)
//   12 FullPAD_Tunnel([6, 9])                      (out:  512)
//   13 FullPAD_Tunnel([4, 10])                     (out:  512)
//   14 FullPAD_Tunnel([8, 11])                     (out: 1024)
//   15 Upsample(-1, scale=2)
//   16 Concat([15, 12], dim=1)                     (out: 1024)
//   17 DSC3k2 n=2 dsc3k=True                       (1024 → 512)
//   18 FullPAD_Tunnel([17, 9])                     (out:  512)
//   19 Upsample(17, scale=2)
//   20 Concat([19, 13], dim=1)                     (out: 1024)
//   21 DSC3k2 n=2 dsc3k=True                       (1024 → 256)
//   22 Conv (10 → 1×1, project hyper map to P3)    (512  → 256)
//   23 FullPAD_Tunnel([21, 22])                    (out:  256)
//   24 Conv k=3 s=2                                (256  → 256)
//   25 Concat([24, 18], dim=1)                     (out: 768)
//   26 DSC3k2 n=2 dsc3k=True                       (768  → 512)
//   27 FullPAD_Tunnel([26, 9])                     (out:  512)
//   28 Conv k=3 s=2                                (512  → 512)
//   29 Concat([28, 14], dim=1)                     (out: 1536)
//   30 DSC3k2 n=2 dsc3k=True                       (1536 → 1024)
//   31 FullPAD_Tunnel([30, 11])                    (out: 1024)
//   32 Detect([23, 27, 31], legacy=false)
//
// All channel counts above are PRE-scaling — runtime channel counts go
// through scale_channels_v13 (round8, clamp to max_channels, * width).
//
// Status: scaffolding in. DSConv / DSBottleneck / DSC3k / DSC3k2 fully
// implemented and parity-tested. HyperACE / FullPAD_Tunnel / FuseModule /
// DownsampleConv / AdaHGConv NOT yet implemented — `forward()` throws.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11.hpp"   // C3 (used as DSC3k base)
#include "yolocpp/models/yolo12.hpp"   // A2C2f
#include "yolocpp/models/yolo8.hpp"    // Conv, Bottleneck, BN scope

namespace yolocpp::models {

// ─── Scales (mirror v12 except m is dropped; iMoonLab ships n/s/l/x) ──
struct Yolo13Scale {
  double depth_multiple;
  double width_multiple;
  int    max_channels;
};

constexpr Yolo13Scale kYolo13n{0.50, 0.25, 1024};
constexpr Yolo13Scale kYolo13s{0.50, 0.50, 1024};
constexpr Yolo13Scale kYolo13l{1.00, 1.00, 512};
constexpr Yolo13Scale kYolo13x{1.00, 1.50, 512};

Yolo13Scale yolo13_scale_from_letter(const std::string& letter);
Yolo13Scale yolo13_scale_from_filename(const std::string& path);

int scale_channels_v13(int c, const Yolo13Scale& s);
int scale_depth_v13(int n, const Yolo13Scale& s);

// ─── DSConv = depthwise k×k → pointwise 1×1 → BN → SiLU ──────────────
//
// Important: BN + activation are applied AFTER the pointwise conv only,
// not after each conv. This is the iMoonLab DSConv shape:
//   forward(x) = act(bn(pw(dw(x))))
//
// State-dict keys per module:
//   dw.weight                 (c_in, 1, k, k)        groups=c_in, bias=False
//   pw.weight                 (c_out, c_in, 1, 1)    bias=False
//   bn.{weight,bias,running_*}                       eps=1e-3
//
struct DSConvImpl : torch::nn::Module {
  torch::nn::Conv2d      dw{nullptr};
  torch::nn::Conv2d      pw{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool                   act_silu = true;
  DSConvImpl(int c_in, int c_out, int k = 3, int s = 1, int p = -1,
             int d = 1, bool act = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DSConv);

// ─── DSBottleneck = DSConv(k1) → DSConv(k2) + optional residual ──────
//
// State-dict keys:
//   cv1.{dw,pw,bn}.<...>      DSConv with k=k1 (default 3)
//   cv2.{dw,pw,bn}.<...>      DSConv with k=k2 (default 5; v13 uses 7 inside DSC3k2)
struct DSBottleneckImpl : torch::nn::Module {
  DSConv cv1{nullptr};
  DSConv cv2{nullptr};
  bool   add = false;
  DSBottleneckImpl(int c1, int c2, bool shortcut = true, double e = 0.5,
                   int k1 = 3, int k2 = 5, int d2 = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DSBottleneck);

// ─── DSC3k — C3 with DSBottleneck inner stack ───────────────────────
//
// Inherits the C3 layout (cv1, cv2, cv3, m) — but cv1/cv2/cv3 are still
// the standard v8 Conv (NOT DSConv); only `m` uses depthwise blocks.
// State-dict keys:
//   cv1.{conv,bn}.<...>       Conv(c1 → c_, 1×1)
//   cv2.{conv,bn}.<...>       Conv(c1 → c_, 1×1)
//   cv3.{conv,bn}.<...>       Conv(2*c_ → c2, 1×1)
//   m.<i>.{cv1,cv2}.<...>     DSBottleneck (e=1.0, k1=3, k2=k2)
struct DSC3kImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  Conv cv3{nullptr};
  torch::nn::Sequential m{nullptr};
  DSC3kImpl(int c1, int c2, int n = 1, bool shortcut = true, double e = 0.5,
            int k1 = 3, int k2 = 5, int d2 = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DSC3k);

// ─── DSC3k2 — C2f-style block with DSBottleneck or DSC3k inner ──────
//
// Same CSP shape as v11's C3k2: cv1 outputs 2*c_inner, cv2 takes
// (2+n)*c_inner. Inner module per index is either a DSC3k(n=2, k1=3, k2=7)
// (when dsc3k=True) or a DSBottleneck(e=1.0, k1=3, k2=7) (when False).
// State-dict keys:
//   cv1.{conv,bn}.<...>       Conv(c1 → 2*c_, 1×1)
//   cv2.{conv,bn}.<...>       Conv((2+n)*c_ → c2, 1×1)
//   m.<i>.<...>               DSC3k or DSBottleneck per dsc3k flag
struct DSC3k2Impl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  int  c_inner = 0;
  bool dsc3k   = false;
  DSC3k2Impl(int c1, int c2, int n = 1, bool dsc3k = false, double e = 0.5,
             bool shortcut = true, int k1 = 3, int k2 = 7, int d2 = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DSC3k2);

// ─── DownsampleConv — AvgPool(2) [+ Conv 1×1 if channel_adjust] ─────
struct DownsampleConvImpl : torch::nn::Module {
  torch::nn::AvgPool2d downsample{nullptr};
  Conv                 channel_adjust_conv{nullptr};
  bool                 channel_adjust = true;
  DownsampleConvImpl(int in_channels, bool channel_adjust = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DownsampleConv);

// ─── FullPAD_Tunnel — gated residual fuser ──────────────────────────
//
// Forward: out = a + gate * b   (gate is a single scalar parameter).
// State-dict key: gate (shape [], scalar).
struct FullPADTunnelImpl : torch::nn::Module {
  torch::Tensor gate;
  FullPADTunnelImpl();
  torch::Tensor forward(torch::Tensor a, torch::Tensor b);
};
TORCH_MODULE(FullPADTunnel);

// ─── FuseModule — multi-scale aligner for HyperACE input ────────────
//
// Takes 3 feature maps [x_hi, x_mid, x_lo] (sorted high→low resolution),
// downsamples x_hi by 2× (avgpool), upsamples x_lo by 2× (nearest), cats
// with x_mid, then 1×1 Conv reduces the cat back to c_in.
//
// channel_adjust=True  → cat is 4*c_in channels (because x_hi at the
//                         pre-pool stage is fed as both pre-pool and post-
//                         pool — see FuseModule sources). conv_out: 4c→c.
// channel_adjust=False → cat is 3*c_in channels. conv_out: 3c→c.
struct FuseModuleImpl : torch::nn::Module {
  torch::nn::AvgPool2d  downsample{nullptr};
  torch::nn::Upsample   upsample{nullptr};
  Conv                  conv_out{nullptr};
  bool                  channel_adjust = true;
  FuseModuleImpl(int c_in, bool channel_adjust);
  torch::Tensor forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(FuseModule);

// ─── AdaHyperedgeGen — context-aware hyperedge participation matrix ──
struct AdaHyperedgeGenImpl : torch::nn::Module {
  torch::Tensor          prototype_base;
  torch::nn::Linear      context_net{nullptr};
  torch::nn::Linear      pre_head_proj{nullptr};
  int    num_heads, num_hyperedges, head_dim;
  std::string context;
  double scaling;
  AdaHyperedgeGenImpl(int node_dim, int num_hyperedges, int num_heads = 4,
                      const std::string& context = "both");
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AdaHyperedgeGen);

// ─── AdaHGConv — vertex→edge→vertex hypergraph conv with residual ───
struct AdaHGConvImpl : torch::nn::Module {
  AdaHyperedgeGen       edge_generator{nullptr};
  torch::nn::Sequential edge_proj{nullptr};
  torch::nn::Sequential node_proj{nullptr};
  AdaHGConvImpl(int embed_dim, int num_hyperedges = 16, int num_heads = 4,
                const std::string& context = "both");
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AdaHGConv);

// ─── AdaHGComputation — wraps AdaHGConv for 4D feature maps ──────────
struct AdaHGComputationImpl : torch::nn::Module {
  AdaHGConv hgnn{nullptr};
  int       embed_dim;
  AdaHGComputationImpl(int embed_dim, int num_hyperedges = 16,
                       int num_heads = 8,
                       const std::string& context = "both");
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AdaHGComputation);

// ─── C3AH — CSP-style block with hypergraph branch ───────────────────
struct C3AHImpl : torch::nn::Module {
  Conv               cv1{nullptr};
  Conv               cv2{nullptr};
  Conv               cv3{nullptr};
  AdaHGComputation   m{nullptr};
  C3AHImpl(int c1, int c2, double e = 1.0, int num_hyperedges = 8,
           const std::string& context = "both");
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C3AH);

// ─── HyperACE — multi-scale hypergraph correlation enhancer ──────────
struct HyperACEImpl : torch::nn::Module {
  Conv                  cv1{nullptr};
  Conv                  cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  FuseModule            fuse{nullptr};
  C3AH                  branch1{nullptr};
  C3AH                  branch2{nullptr};
  int                   c_inner = 0;
  HyperACEImpl(int c1, int c2, int n = 1, int num_hyperedges = 8,
               bool dsc3k = true, bool shortcut = false, double e1 = 0.5,
               double e2 = 1.0, const std::string& context = "both",
               bool channel_adjust = true);
  torch::Tensor forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(HyperACE);

// ─── V13AAttn — area-windowed attention, iMoonLab variant ───────────
//
// Differs from v12's AAttn:
//   - Two separate Convs `qk` (out=2C) and `v` (out=C), NOT a fused qkv.
//   - `pe` is depthwise k=5 (v12 uses k=7).
//   - pe operates on `v` (after the v Conv), not on the qkv stream.
//   - Final output: proj(attn_out + pe(v_4d)).
//
// num_heads supplied externally; head_dim = dim / num_heads.
struct V13AAttnImpl : torch::nn::Module {
  Conv qk{nullptr};
  Conv v{nullptr};
  Conv proj{nullptr};
  Conv pe{nullptr};
  int num_heads, head_dim, area;
  V13AAttnImpl(int dim, int num_heads, int area = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(V13AAttn);

// ─── V13ABlock — same residual shape as v12 ABlock but uses V13AAttn ──
struct V13ABlockImpl : torch::nn::Module {
  V13AAttn              attn{nullptr};
  torch::nn::Sequential mlp{nullptr};
  V13ABlockImpl(int dim, int num_heads, double mlp_ratio = 1.2, int area = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(V13ABlock);

// ─── V13A2C2f — same shape as v12's A2C2f, holds V13ABlock pairs ──────
struct V13A2C2fImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  bool a2 = true;
  bool residual = false;
  int  c_inner = 0;
  torch::Tensor gamma;
  bool has_gamma = false;
  V13A2C2fImpl(int c1, int c2, int n = 1, bool a2 = true, int area = 1,
               bool residual = false, double mlp_ratio = 2.0, double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(V13A2C2f);

// ─── Yolo13Detect — full network ─────────────────────────────────────
struct Yolo13DetectImpl : torch::nn::Module {
  Yolo13Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};   // 33 entries
  std::vector<double>   stride;

  Yolo13DetectImpl(Yolo13Scale s, int nc);

  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo13Detect);

}  // namespace yolocpp::models
