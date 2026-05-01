#pragma once
//
// YOLO12 — Tian et al., "YOLO12: Attention-Centric Real-Time Object
// Detectors" (Feb 2025), shipped through the upstream asset host.
//
// Architectural summary vs. v11:
//   - **No SPPF, no C2PSA.** v11's deep-stage feature mixing is replaced
//     by `A2C2f` blocks at backbone layers 6 and 8.
//   - **A2C2f** ("Area-Attention" CSP-Fast): a CSP-style block whose
//     inner module is a stack of two `ABlock`s (when `a2=True`) or a
//     single `C3k(c_, c_, n=2)` (when `a2=False`). cv1 outputs `c_inner`
//     channels (NOT `2*c_inner` like C2f / C3k2); cv2 takes
//     `(1 + n) * c_inner` channels.
//   - **ABlock**: residual `x + mlp(x + attn(x))`. mlp is two 1×1 Convs.
//     The attention is `AAttn`, an area-windowed multi-head self-
//     attention with a 7×7 depthwise positional encoding on V.
//   - **Head**: standard Detect with v11's nested DWConv cv3 form
//     (legacy=false). cv2 stays Conv→Conv→Conv2d.
//
// The 22-layer module list:
//   0..1   Conv stem
//   2,4    C3k2 (c3k=False, e=0.25)
//   6      A2C2f (a2=True, area=4) at backbone P4-stage
//   8      A2C2f (a2=True, area=1) at backbone P5-stage
//   9..14  PAN-up (Upsample + Concat + A2C2f a2=False)
//   15..20 PAN-down (Conv + Concat + A2C2f / C3k2-c3k=True)
//   21     Detect (legacy=false)
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11.hpp"   // reuse Conv, Bottleneck, C3k, C3k2
#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::models {

// ─── Scales (mirror v11; A2C2f at deep layers behaves the same way) ───
struct Yolo12Scale {
  double depth_multiple;
  double width_multiple;
  int    max_channels;
};

constexpr Yolo12Scale kYolo12n{0.50, 0.25, 1024};
constexpr Yolo12Scale kYolo12s{0.50, 0.50, 1024};
constexpr Yolo12Scale kYolo12m{0.50, 1.00,  512};
constexpr Yolo12Scale kYolo12l{1.00, 1.00,  512};
constexpr Yolo12Scale kYolo12x{1.00, 1.50,  512};

Yolo12Scale yolo12_scale_from_letter(const std::string& letter);
Yolo12Scale yolo12_scale_from_filename(const std::string& path);

int scale_channels_v12(int c, const Yolo12Scale& s);
int scale_depth_v12(int n, const Yolo12Scale& s);

// ─── AAttn — area-windowed multi-head self-attention ──────────────────────
//
// `area` ∈ {1, 4} in shipped v12 (only L6 uses 4; everything else uses 1).
// When area > 1, the spatial dim N=H*W is split into `area` regions and
// attention is applied independently within each region (windowed
// attention) — equivalent to torch's "area attention".
//
// State-dict keys (per module):
//   qkv.{conv,bn}.{weight,bias,running_*}      (Conv, act=False)
//   proj.{conv,bn}.{weight,bias,running_*}     (Conv, act=False)
//   pe.{conv,bn}.{weight,bias,running_*}       (depthwise 7×7, act=False;
//                                               pe.conv has bias=True)
struct AAttnImpl : torch::nn::Module {
  Conv qkv{nullptr};
  Conv proj{nullptr};
  Conv pe{nullptr};
  int  num_heads, head_dim, area;
  double scale;
  AAttnImpl(int dim, int num_heads, int area = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AAttn);

// ─── ABlock — attention + MLP with double residual ────────────────────────
//
// Forward:  y = x + attn(x)
//           z = y + mlp(y)
// State-dict keys:
//   attn.{...}                 (AAttn)
//   mlp.0.{conv,bn}.<...>      (Conv, act=True default)
//   mlp.1.{conv,bn}.<...>      (Conv, act=False)
struct ABlockImpl : torch::nn::Module {
  AAttn attn{nullptr};
  torch::nn::Sequential mlp{nullptr};
  ABlockImpl(int dim, int num_heads, double mlp_ratio = 1.2, int area = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ABlock);

// ─── A2C2f — "Area-Attention C2f" CSP block ───────────────────────────────
//
// CSP shape:
//   cv1 (1×1, c1 → c_inner)
//   m   (ModuleList of n entries)
//   cv2 (1×1, (1 + n) * c_inner → c2)
// Forward:  y = [cv1(x)]; for i: y.append(m[i](y[-1])); cv2(cat(y)).
//
// `a2=True`  → m[i] = Sequential(ABlock × 2)
// `a2=False` → m[i] = C3k(c_inner, c_inner, n=2)   (matches v11 C3k2 inner
//                                                    when c3k=True)
//
// `residual=True` adds an outer x → y skip when c1 == c2 (used at the
// gamma-scaled deep layers in newer v12 variants; we expose the flag but
// the shipped v12<n,s,m,l,x>.pt all have residual=False).
struct A2C2fImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  bool a2 = true;
  bool residual = false;
  int  c_inner = 0;
  // Per-channel learned scale on the residual path. Registered as a
  // parameter named `gamma` (shape [c2]) only when residual=true && c1==c2.
  // The shipped v12l/v12x weights ship with `model.<i>.gamma`; without this
  // gate the residual `x + y` produces saturated cls outputs (we caught
  // this at predict-time on v12l: 300 detections at conf=0.25).
  torch::Tensor gamma;
  bool          has_gamma = false;
  A2C2fImpl(int c1, int c2, int n = 1, bool a2 = true, int area = 1,
            bool residual = false, double mlp_ratio = 2.0, double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(A2C2f);

// ─── Yolo12Detect ─────────────────────────────────────────────────────────
struct Yolo12DetectImpl : torch::nn::Module {
  Yolo12Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo12DetectImpl(Yolo12Scale s, int nc);

  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo12Detect);

}  // namespace yolocpp::models
