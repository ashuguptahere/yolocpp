#pragma once
//
// YOLO11 — upstream official (Sept 2024). Filename convention: `yolo11<n/s/m/l/x>.pt`
// (no 'v').
//
// What's new vs. YOLO8:
//   - C2f → **C3k2**: same outer CSP topology (cv1 split + n inner blocks +
//     cv2 merge) but each inner module can be either a plain Bottleneck
//     (c3k=False) or a nested C3k block (c3k=True). C3k itself is the
//     classic C3 with 3×3 inner Bottleneck kernels.
//   - **C2PSA** added at the deepest backbone stage (after SPPF). Splits
//     channels two-way; the second half passes through n PSABlocks, each of
//     which is an Attention(qkv, depthwise positional encoding, proj) plus
//     an FFN with shortcut.
//   - **Detect head's cv3 branch** swaps the legacy (Conv→Conv→Conv2d)
//     stack for a depthwise-separable nested form
//     (DWConv→Conv) → (DWConv→Conv) → Conv2d.  cv2 (regression) is unchanged.
//
// Scales (from yolo11.yaml):
//   n: depth=0.50, width=0.25, max_channels=1024
//   s: depth=0.50, width=0.50, max_channels=1024
//   m: depth=0.50, width=1.00, max_channels=512
//   l: depth=1.00, width=1.00, max_channels=512
//   x: depth=1.00, width=1.50, max_channels=512
//
// State-dict layout matches upstream exactly so that yolo11<N>.pt loads
// without remapping. The 24-module list:
//   0-9   backbone: Conv, Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF
//   10    C2PSA
//   11-22 head: Upsample / Concat / C3k2 (× the v8 PAN topology)
//   23    Detect (legacy=false → DWConv-Conv cv3)
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo8.hpp"  // reuse Conv, DWConv, Bottleneck, SPPF, Detect

namespace yolocpp::models {

// ─── Scale (separate type from Yolo8Scale to keep call sites explicit) ────
struct Yolo11Scale {
  double depth_multiple;
  double width_multiple;
  int    max_channels;
};

constexpr Yolo11Scale kYolo11n{0.50, 0.25, 1024};
constexpr Yolo11Scale kYolo11s{0.50, 0.50, 1024};
constexpr Yolo11Scale kYolo11m{0.50, 1.00,  512};
constexpr Yolo11Scale kYolo11l{1.00, 1.00,  512};
constexpr Yolo11Scale kYolo11x{1.00, 1.50,  512};

// Map a single-letter scale spec (n/s/m/l/x) to a Yolo11Scale.
Yolo11Scale yolo11_scale_from_letter(const std::string& letter);

// Helpers (mirror v8's scale_channels / scale_depth but parameterised on Yolo11Scale).
int scale_channels_v11(int c, const Yolo11Scale& s);
int scale_depth_v11(int n, const Yolo11Scale& s);

// ─── C3k = C3 with 3×3 inner Bottlenecks ───────────────────────────────────
// Internals: cv1 (1×1, c1→c_), cv2 (1×1, c1→c_), cv3 (1×1, 2*c_→c2),
// m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, k=(3,3), e=1.0) for _ in range(n)])
struct C3kImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  Conv cv3{nullptr};
  // Inner Bottlenecks held in a ModuleList (vs Sequential) so we can
  // iterate them by exact concrete type and avoid the AnyModule chain that
  // Sequential uses internally.
  torch::nn::ModuleList m{nullptr};
  C3kImpl(int c1, int c2, int n, bool shortcut = true, int g = 1,
          double e = 0.5, int k = 3);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C3k);

// ─── C3k2 = C2f variant whose inner blocks are Bottleneck or C3k ──────────
// State-dict naming matches v8 C2f's: cv1.{conv,bn}, cv2.{conv,bn},
// m.<i>.{cv1,cv2,cv3,m...}.
struct C3k2Impl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  bool   c3k = false;        // true → m[i] is C3k, false → m[i] is Bottleneck
  int    c_inner = 0;        // = int(c2 * e)
  C3k2Impl(int c1, int c2, int n = 1, bool c3k = false, double e = 0.5,
          int g = 1, bool shortcut = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C3k2);

// ─── PSA Attention (qkv conv + depthwise positional encoding + proj conv) ─
// Notes (matching the upstream `nn/modules/block.py::Attention`):
//   num_heads = c // 64 (set by C2PSA at construction), key_dim = head_dim*0.5,
//   qkv: Conv(dim, dim + 2*nh*key_dim, 1, act=False),
//   pe : Conv(dim, dim, 3, g=dim, act=False)  // depthwise pos-encoding,
//   proj: Conv(dim, dim, 1, act=False).
// Forward (N=H*W tokens):
//   qkv → split into (q, k, v) along the channel dim,
//   attn = softmax((qᵀ@k) * scale) where scale = key_dim^-0.5,
//   x = (v @ attnᵀ).reshape(B,C,H,W) + pe(v.reshape(B,C,H,W)),
//   x = proj(x).
struct PSAAttentionImpl : torch::nn::Module {
  Conv qkv{nullptr};
  Conv proj{nullptr};
  Conv pe{nullptr};
  int num_heads, head_dim, key_dim;
  double scale;
  PSAAttentionImpl(int dim, int num_heads, double attn_ratio = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PSAAttention);

// ─── PSABlock = Attention + FFN, both with shortcut (default) ──────────────
struct PSABlockImpl : torch::nn::Module {
  PSAAttention attn{nullptr};
  torch::nn::Sequential ffn{nullptr};
  bool add = true;
  PSABlockImpl(int c, double attn_ratio = 0.5, int num_heads = 4,
               bool shortcut = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PSABlock);

// ─── C2PSA: CSP-style split with n PSABlocks on the second branch ─────────
// Internals: cv1 (1×1, c1→2*c_), cv2 (1×1, 2*c_→c1=c2), m = Sequential of
// n PSABlocks operating on the second half.
struct C2PSAImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::Sequential m{nullptr};
  int c = 0;     // = int(c1 * e); per-branch channels
  C2PSAImpl(int c1, int c2, int n = 1, double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C2PSA);

// ─── Whole detection model ────────────────────────────────────────────────
//
// model[0..9]   backbone (Conv/C3k2/SPPF)
// model[10]     C2PSA
// model[11..22] head (Upsample/Concat/C3k2 — same PAN topology as v8)
// model[23]     Detect (legacy=false)
struct Yolo11DetectImpl : torch::nn::Module {
  Yolo11Scale scale;
  int          nc;

  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo11DetectImpl(Yolo11Scale s, int nc);

  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo11Detect);

// Auto-detect scale from a yolo11<scale>.pt filename. Returns kYolo11n on
// failure (callers should override).
Yolo11Scale yolo11_scale_from_filename(const std::string& path);

}  // namespace yolocpp::models
