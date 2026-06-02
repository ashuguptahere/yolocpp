#pragma once
//
// YOLO6 (Meituan v3.0 / release 0.4.0 baseline) — DEPLOY FORM.
//
// Backbone : EfficientRep (RepVGG → fused single Conv at deploy)
// Neck     : RepBiFPANNeck (BiFusion blocks with ConvTranspose2d upsample)
// Head     : EffiDeHead — anchor-free decoupled, DFL with reg_max=16
//            (17 bins per side, 4 sides → reg_preds_dist has 4*17=68 ch).
//            Inference path uses reg_preds_dist + DFL projection to
//            produce 4-channel ltrb distances; the auxiliary direct
//            `reg_preds` (4 ch) is for training only.
// Activation: ReLU throughout.
//
// We ship the DEPLOY form: each RepVGG block is a single Conv2d(bias=true)
// + ReLU, BN already fused in. Conversion from Meituan's train-form .pt
// (rbr_dense + rbr_1x1 + rbr_identity) is handled by
// `serialization::convert_yolov6_pt(...)`, which applies RepVGG
// re-parameterization to collapse the three branches into one 3×3 kernel
// + bias before saving.
//
// Currently supports yolo6s only (the smallest "standard" variant).
// n/m/l have minor topology differences (m/l swap the neck for
// CSPRepBiFPANNeck) and will be added alongside their state-dict
// quirks.
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::models {

// ─── Yolo6Variant — chooses backbone topology + activation regime ────────
//
// Standard (n/s/m/l): EfficientRep (RepBlock for n/s, BepC3 for m/l) +
//   RepBiFPANNeck. n/s/m use ReLU; l uses SiLU (training_mode=conv_silu).
// MBLA (s_mbla / m_mbla / l_mbla / x_mbla): CSPBepBackbone + CSPRepBiFPANNeck
//   with MBLABlock as the stage block. All MBLA scales use SiLU
//   (training_mode=conv_silu) and ConvBNSiLU as the basic block. csp_e=0.5.
enum class Yolo6Variant { Standard, MBLA };

struct Yolo6Scale {
  double depth_multiple;
  double width_multiple;
  Yolo6Variant variant = Yolo6Variant::Standard;
};
constexpr Yolo6Scale kYolo6n     {0.33, 0.25, Yolo6Variant::Standard};
constexpr Yolo6Scale kYolo6s     {0.33, 0.50, Yolo6Variant::Standard};
constexpr Yolo6Scale kYolo6m     {0.60, 0.75, Yolo6Variant::Standard};
constexpr Yolo6Scale kYolo6l     {1.00, 1.00, Yolo6Variant::Standard};
constexpr Yolo6Scale kYolo6s_mbla{0.50, 0.50, Yolo6Variant::MBLA};
constexpr Yolo6Scale kYolo6m_mbla{0.50, 0.75, Yolo6Variant::MBLA};
constexpr Yolo6Scale kYolo6l_mbla{0.50, 1.00, Yolo6Variant::MBLA};
constexpr Yolo6Scale kYolo6x_mbla{1.00, 1.00, Yolo6Variant::MBLA};

// P6 flag is passed as a separate ctor argument to Yolo6Impl (not part of
// Yolo6Scale, since the multipliers are identical to the P5 variants
// — only the head topology differs).

// ─── ConvBNReLU = Conv2d + BN + (ReLU | SiLU) ───────────────────────────
//
// Default activation is ReLU (Meituan's "SimConv"). When `V6ActScope` is
// active — Yolo6Impl pushes one when constructing the `l` scale, whose
// upstream training_mode is `conv_silu` — the activation switches to
// SiLU. Captured at construction time so forward stays a simple branch
// on the per-instance flag.
struct ConvBNReLUImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool                   use_silu = false;
  // Inference-time Conv+BN fusion (#95B-extended). Same idea as
  // yolo8.cpp's ConvImpl::fuse. After fuse(), forward skips BN.
  bool                   fused = false;
  torch::Tensor          fused_weight;
  torch::Tensor          fused_bias;
  void fuse();
  ConvBNReLUImpl(int c_in, int c_out, int k = 1, int s = 1, int p = -1, int g = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvBNReLU);

// ─── V6ActScope — RAII toggle for ConvBNReLU activation ────────────────
struct V6ActScope {
  bool prev;
  explicit V6ActScope(bool use_silu);
  ~V6ActScope();
};

// ─── RepConv (deploy form) — fused 3×3 Conv2d with bias + ReLU ───────────
struct RepConvImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  RepConvImpl(int c_in, int c_out, int k = 3, int s = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepConv);

// ─── RepBlock — conv1 RepConv + N-1 stacked RepConvs in `block` ──────────
struct RepBlockImpl : torch::nn::Module {
  RepConv               conv1{nullptr};
  torch::nn::ModuleList block{nullptr};
  RepBlockImpl(int c_in, int c_out, int n);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepBlock);

// ─── BottleRep — two basic blocks back-to-back with optional shortcut ────
//
// Each "basic block" is either a RepConv (deploy form: 3×3 Conv2d + bias
// + ReLU; used by v6m) or a ConvBNReLU (used by v6l). Both register
// under the same child names "conv1" / "conv2" so upstream key paths
// (`<prefix>.conv1.<sub>` and `.conv2.<sub>`) match either way.
//
// BottleRep also has a learned `alpha` scalar (Parameter shape [1]) on
// the shortcut path: `out = block2(block1(x)) + alpha * x`. This is
// upstream's `weight=True` mode used by both v6m and v6l.
struct BottleRepImpl : torch::nn::Module {
  RepConv    rep_conv1{nullptr}, rep_conv2{nullptr};
  ConvBNReLU cbr_conv1{nullptr}, cbr_conv2{nullptr};
  torch::Tensor alpha;          // learned shortcut weight, shape [1]
  bool       use_repconv = true;
  bool       add         = true;
  BottleRepImpl(int c_in, int c_out, bool use_repconv);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(BottleRep);

// ─── BottleRep3 — three basic blocks back-to-back with optional shortcut ──
//
// Used by MBLABlock. Each "basic block" is a ConvBNSiLU (3×3) when MBLA's
// `block=ConvBNSiLU` upstream (training_mode=conv_silu, which is the case
// for all four MBLA variants). The learned `alpha` scalar (Parameter
// shape [1]) gates the shortcut: `out = conv3(conv2(conv1(x))) + alpha*x`
// when c_in == c_out.
struct BottleRep3Impl : torch::nn::Module {
  ConvBNReLU    conv1{nullptr};
  ConvBNReLU    conv2{nullptr};
  ConvBNReLU    conv3{nullptr};
  torch::Tensor alpha;            // shape [1]
  bool          add = true;
  BottleRep3Impl(int c_in, int c_out);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(BottleRep3);

// ─── MBLABlock — Multi-Branch Linear Activation Block (used by MBLA scales) ──
//
// Topology (Meituan v6 layers/common.py:653):
//   internal_n = max(n // 2, 1)
//   if internal_n == 1: n_list = [0, 1]
//   else:               n_list = [0, extra, internal_n]   # extra = largest power-of-2 < internal_n
//   branch_num = len(n_list)
//   c_ = int(c_out * e)
//   cv1: c_in → branch_num*c_  (1×1 ConvBNSiLU)
//   cv2: (sum(n_list)+branch_num)*c_ → c_out  (1×1 ConvBNSiLU)
//   m  : ModuleList of `branch_num - 1` Sequentials, where m[i] holds
//        n_list[i+1] BottleRep3s.
//
// Forward (with branch_num=3 example):
//   y0,y1,y2 = cv1(x).split(c_, dim=1)
//   all = [y0]
//   for j in 0..branch_num-2:
//       all.append(y[j+1])
//       cur = y[j+1]
//       for k in 0..n_list[j+1]-1:
//           cur = m[j][k](cur)
//           all.append(cur)
//   return cv2(cat(all, 1))
struct MBLABlockImpl : torch::nn::Module {
  ConvBNReLU             cv1{nullptr};
  ConvBNReLU             cv2{nullptr};
  torch::nn::ModuleList  m{nullptr};      // branch_num - 1 entries; each is
                                           // a Sequential of n_list[i+1]
                                           // BottleRep3s.
  int                    c_inner = 0;     // c_ from upstream (= c_out * e)
  std::vector<int>       n_list;          // [0, ..., internal_n]
  int                    branch_num = 0;
  MBLABlockImpl(int c_in, int c_out, int n, double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MBLABlock);

// ─── RepBlockBR — RepBlock variant with BottleRep inner blocks ───────────
struct RepBlockBRImpl : torch::nn::Module {
  BottleRep             conv1{nullptr};
  torch::nn::ModuleList block{nullptr};
  bool                  use_repconv = true;
  RepBlockBRImpl(int c_in, int c_out, int n, bool use_repconv);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepBlockBR);

// ─── BepC3 — CSP-wrapped RepBlockBR (used by v6m / v6l) ─────────────────
//
// cv1: c_in → c_ (1×1 ConvBNReLU)         CSP "main" path
// cv2: c_in → c_ (1×1 ConvBNReLU)         CSP "shortcut" path
// cv3: 2*c_ → c_out (1×1 ConvBNReLU)
// m  : RepBlockBR(c_, c_, n, use_repconv)
// where c_ = int(c_out * e). e=2/3 for v6m, 1/2 for v6l.
struct BepC3Impl : torch::nn::Module {
  ConvBNReLU cv1{nullptr};
  ConvBNReLU cv2{nullptr};
  ConvBNReLU cv3{nullptr};
  RepBlockBR m{nullptr};
  BepC3Impl(int c_in, int c_out, int n, bool use_repconv, double e);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(BepC3);

// ─── CSPSPPF — Meituan's CSP-wrapped SPPF (5/9/13 maxpools) ──────────────
//
// Topology (from yolov6/layers/common.py):
//   y1 = cv2(cv1(x))            # main path
//   p1 = m(y1)
//   p2 = m(p1)
//   p3 = m(p2)
//   y_main = cv6(cv5(cat(y1, p1, p2, p3)))
//   y_short = cv4(cv3(x))       # CSP shortcut path; cv3 is 3×3
//   out = cv7(cat(y_main, y_short))
//
// Looking at upstream shapes for yolov6s (c=512):
//   cv1: 512 → 256 (1×1)
//   cv2: 256 → 256 (1×1)        wait — recheck shapes
//   actually we observed: cv1=[256,512,1,1], cv2=[256,512,1,1] — both 512→256
struct CSPSPPFImpl : torch::nn::Module {
  ConvBNReLU           cv1{nullptr};
  ConvBNReLU           cv2{nullptr};
  ConvBNReLU           cv3{nullptr};
  ConvBNReLU           cv4{nullptr};
  ConvBNReLU           cv5{nullptr};
  ConvBNReLU           cv6{nullptr};
  ConvBNReLU           cv7{nullptr};
  torch::nn::MaxPool2d m{nullptr};
  CSPSPPFImpl(int c_in, int c_out, int k = 5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CSPSPPF);

// ─── SPPFModule — inner of SimSPPF (used by v6m / v6l) ──────────────────
struct SPPFModuleImpl : torch::nn::Module {
  ConvBNReLU           cv1{nullptr};
  ConvBNReLU           cv2{nullptr};
  torch::nn::MaxPool2d m{nullptr};
  SPPFModuleImpl(int c_in, int c_out, int k = 5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPFModule);

// ─── SimSPPF — Meituan-style SPPF wrapper (used by v6m / v6l) ────────────
struct SimSPPFImpl : torch::nn::Module {
  SPPFModule sppf{nullptr};
  SimSPPFImpl(int c_in, int c_out, int k = 5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SimSPPF);

// ─── Transpose2d wrapper (so state_dict key matches Meituan's
//     `upsample_transpose.weight` / `.bias`) ─────────────────────────────
struct TransposeImpl : torch::nn::Module {
  torch::nn::ConvTranspose2d upsample_transpose{nullptr};
  TransposeImpl(int c_in, int c_out);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Transpose);

// ─── BiFusionBlock — Meituan's bi-directional fusion node ────────────────
//
// Three inputs at three resolutions: small (high-stride) `xs`, mid (this
// scale) `xm`, large (low-stride) `xl`. Routes:
//   xs  = upsample(xs)          (transposed conv 2×2 stride 2)
//   xm' = downsample(xm)        (3×3 stride 2, RepConv-style ConvBNReLU)
//   xl  = cv1(xl)
//   xm  = cv2(xm)
//   out = cv3(cat(xs, xm, xl_down))   shapes: [c, c, c] → cv3 inputs 3*c.
struct BiFusionBlockImpl : torch::nn::Module {
  ConvBNReLU cv1{nullptr};
  ConvBNReLU cv2{nullptr};
  ConvBNReLU cv3{nullptr};
  Transpose  upsample{nullptr};
  ConvBNReLU downsample{nullptr};
  BiFusionBlockImpl(int c_in_lateral_to_p4, int c_in_lateral_to_p3,
                    int c_mid, int c_out);
  // x = (mid_at_this_scale, lower_resolution_after_reduce, higher_resolution_lateral)
  torch::Tensor forward(torch::Tensor mid, torch::Tensor lower_reduced,
                        torch::Tensor higher_lateral);
};
TORCH_MODULE(BiFusionBlock);

// ─── EffiDeHead (decoupled detect head, anchor-free, DFL) ────────────────
//
// Per-scale: stem (1×1 to keep ch), then split:
//   cls branch: cls_convs (3×3) → cls_preds (1×1) → nc channels
//   reg branch: reg_convs (3×3) → reg_preds_dist (1×1) → 4*(reg_max+1) ch
//                                → reg_preds      (1×1) → 4 ch (training)
//
// Inference: project reg_preds_dist via proj_conv (DFL) to produce 4-ch
// ltrb distances; then dist2bbox + cat with cls_sigmoid → [B, 4+nc, A].
struct EffiDeHeadImpl : torch::nn::Module {
  int nc;
  int reg_max;          // = 16; bins per side = reg_max + 1 = 17
  // For v6n/s: reg_preds is 4-ch direct ltrb (used at eval). reg_preds_dist
  // is the 68-ch DFL distillation target (training only).
  // For v6m/l: reg_preds is 68-ch DFL (used at eval). No separate
  // reg_preds_dist (it doesn't exist upstream for these scales).
  bool dfl_eval = false;

  // Per-scale (length 3 each: P3, P4, P5)
  torch::nn::ModuleList stems{nullptr};
  torch::nn::ModuleList cls_convs{nullptr};
  torch::nn::ModuleList reg_convs{nullptr};
  torch::nn::ModuleList cls_preds{nullptr};
  torch::nn::ModuleList reg_preds_dist{nullptr};   // n/s only
  torch::nn::ModuleList reg_preds{nullptr};

  torch::Tensor      proj;        // [reg_max + 1] — buffer; arange
  torch::nn::Conv2d  proj_conv{nullptr};

  // Channels per scale. Use 3 levels (P3, P4, P5) for standard v6 or
  // 4 levels (P3, P4, P5, P6) for the P6 variants (#24).
  int num_layers = 3;
  EffiDeHeadImpl(int nc, int c_p3, int c_p4, int c_p5, int reg_max = 16,
                 bool dfl_eval = false);
  EffiDeHeadImpl(int nc, const std::vector<int>& chans, int reg_max,
                 bool dfl_eval);
  std::vector<torch::Tensor> forward_eval_per_scale(
      torch::Tensor p3, torch::Tensor p4, torch::Tensor p5,
      const std::vector<int>& strides, int img_h, int img_w);
  // N-level form (P6: feats has 4 elements).
  std::vector<torch::Tensor> forward_eval_per_scale_n(
      const std::vector<torch::Tensor>& feats,
      const std::vector<int>& strides);
  // Training-mode forward: returns raw per-scale [B, 4*bins + nc, H, W]
  // feature maps suitable for V6DetectionLoss. Uses the DFL `reg_preds_dist`
  // branch when present (n/s/n6/s6 — KD distillation target was training-
  // only upstream) or the `reg_preds` branch directly (m/l/m6/l6).
  std::vector<torch::Tensor> forward_train_per_scale_n(
      const std::vector<torch::Tensor>& feats);
};
TORCH_MODULE(EffiDeHead);

// ─── Yolo6 model (top-level, yolo6s deploy form) ─────────────────────────
struct Yolo6Impl : torch::nn::Module {
  // Public scale + nc fields for trainer EMA construction.
  // Declared first so the (scale, nc, reg_max, p6) initializer list runs
  // in declaration order (`scale` before `nc/reg_max/is_p6`).
  Yolo6Scale scale = kYolo6s;
  int nc;
  int reg_max;
  // Per-level strides — populated lazily on first forward via the
  // hardcoded [8,16,32] (P5) or [8,16,32,64] (P6) values used inside
  // forward(). Exposed for the trainer (which passes strides to
  // V6DetectionLoss) and for loss/val pipelines that need them.
  std::vector<double> stride;

  // Backbone (EfficientRep) — for n/s the inner blocks are RepBlocks of
  // RepConv; for m/l they're BepC3 (CSP-wrapped) of BottleRep. The
  // ERBlock_N_down stems are RepConv (deploy: single 3×3 with bias) for
  // n/s/m, and ConvBNReLU for l (no fusion needed). Only one of
  // {ERBlock_N_block_rb, ERBlock_N_block_bep} is registered per scale —
  // they share the state-dict slot name "ERBlock_N_block".
  struct BackboneImpl : torch::nn::Module {
    // Stem & downsamples — alternative types per scale.
    RepConv     stem_rep{nullptr};
    ConvBNReLU  stem_cbr{nullptr};
    RepConv     ERBlock_2_down_rep{nullptr};
    ConvBNReLU  ERBlock_2_down_cbr{nullptr};
    RepConv     ERBlock_3_down_rep{nullptr};
    ConvBNReLU  ERBlock_3_down_cbr{nullptr};
    RepConv     ERBlock_4_down_rep{nullptr};
    ConvBNReLU  ERBlock_4_down_cbr{nullptr};
    RepConv     ERBlock_5_down_rep{nullptr};
    ConvBNReLU  ERBlock_5_down_cbr{nullptr};
    // Inner blocks — RepBlock for n/s, BepC3 for m/l, MBLABlock for *_mbla.
    RepBlock    ERBlock_2_block_rb{nullptr};
    BepC3       ERBlock_2_block_bep{nullptr};
    MBLABlock   ERBlock_2_block_mbla{nullptr};
    RepBlock    ERBlock_3_block_rb{nullptr};
    BepC3       ERBlock_3_block_bep{nullptr};
    MBLABlock   ERBlock_3_block_mbla{nullptr};
    RepBlock    ERBlock_4_block_rb{nullptr};
    BepC3       ERBlock_4_block_bep{nullptr};
    MBLABlock   ERBlock_4_block_mbla{nullptr};
    RepBlock    ERBlock_5_block_rb{nullptr};
    BepC3       ERBlock_5_block_bep{nullptr};
    MBLABlock   ERBlock_5_block_mbla{nullptr};
    CSPSPPF     ERBlock_5_cspsppf{nullptr};   // for n/s
    SimSPPF     ERBlock_5_simsppf{nullptr};   // for m/l/MBLA (registered
                                               // under the same slot name
                                               // "ERBlock_5_cspsppf")
    // Construction flags.
    bool        use_rep_blocks = true;   // false → BepC3 / MBLABlock
    bool        use_repconv    = true;   // false → ConvBNReLU stems (l/MBLA)
    bool        use_mbla       = false;  // true → MBLABlock inner blocks
    BackboneImpl(int c0, int c1, int c2, int c3, int c4,
                 int n2, int n3, int n4, int n5,
                 bool use_rep_blocks, bool use_repconv, double bep_e,
                 bool use_mbla = false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor x);
  };

  struct NeckImpl : torch::nn::Module {
    ConvBNReLU    reduce_layer0{nullptr};
    BiFusionBlock Bifusion0{nullptr};
    RepBlock      Rep_p4_rb{nullptr};
    BepC3         Rep_p4_bep{nullptr};
    MBLABlock     Rep_p4_mbla{nullptr};
    ConvBNReLU    reduce_layer1{nullptr};
    BiFusionBlock Bifusion1{nullptr};
    RepBlock      Rep_p3_rb{nullptr};
    BepC3         Rep_p3_bep{nullptr};
    MBLABlock     Rep_p3_mbla{nullptr};
    ConvBNReLU    downsample2{nullptr};
    RepBlock      Rep_n3_rb{nullptr};
    BepC3         Rep_n3_bep{nullptr};
    MBLABlock     Rep_n3_mbla{nullptr};
    ConvBNReLU    downsample1{nullptr};
    RepBlock      Rep_n4_rb{nullptr};
    BepC3         Rep_n4_bep{nullptr};
    MBLABlock     Rep_n4_mbla{nullptr};
    bool          use_rep_blocks = true;
    bool          use_repconv    = true;
    bool          use_mbla       = false;
    NeckImpl(int c2, int c3, int c4, int n_neck,
             bool use_rep_blocks, bool use_repconv, double bep_e,
             bool use_mbla = false);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor p3, torch::Tensor p4, torch::Tensor p5);
  };

  // ─── P6 backbone — 6-stage EfficientRep6 / CSPBepBackbone_P6 ──────────
  struct BackboneP6Impl : torch::nn::Module {
    // Stems & inter-stage downsamples. P6 has 6 levels: stem + ERBlock_2..6.
    RepConv     stem_rep{nullptr};            ConvBNReLU  stem_cbr{nullptr};
    RepConv     ERBlock_2_down_rep{nullptr};  ConvBNReLU  ERBlock_2_down_cbr{nullptr};
    RepConv     ERBlock_3_down_rep{nullptr};  ConvBNReLU  ERBlock_3_down_cbr{nullptr};
    RepConv     ERBlock_4_down_rep{nullptr};  ConvBNReLU  ERBlock_4_down_cbr{nullptr};
    RepConv     ERBlock_5_down_rep{nullptr};  ConvBNReLU  ERBlock_5_down_cbr{nullptr};
    RepConv     ERBlock_6_down_rep{nullptr};  ConvBNReLU  ERBlock_6_down_cbr{nullptr};
    // Inner blocks — RepBlock for n6/s6, BepC3 for m6/l6.
    RepBlock    ERBlock_2_block_rb{nullptr};  BepC3 ERBlock_2_block_bep{nullptr};
    RepBlock    ERBlock_3_block_rb{nullptr};  BepC3 ERBlock_3_block_bep{nullptr};
    RepBlock    ERBlock_4_block_rb{nullptr};  BepC3 ERBlock_4_block_bep{nullptr};
    RepBlock    ERBlock_5_block_rb{nullptr};  BepC3 ERBlock_5_block_bep{nullptr};
    RepBlock    ERBlock_6_block_rb{nullptr};  BepC3 ERBlock_6_block_bep{nullptr};
    // SPP — CSPSPPF (n6/s6) or SimSPPF (m6/l6) — at ERBlock_6 (NOT _5).
    CSPSPPF     ERBlock_6_cspsppf{nullptr};
    SimSPPF     ERBlock_6_simsppf{nullptr};
    bool        use_rep_blocks = true;
    bool        use_repconv    = true;
    BackboneP6Impl(int c0, int c1, int c2, int c3, int c4, int c5,
                   int n2, int n3, int n4, int n5, int n6,
                   bool use_rep_blocks, bool use_repconv, double bep_e);
  };

  // ─── P6 neck — RepBiFPANNeck6 / CSPRepBiFPANNeck_P6 ───────────────────
  struct NeckP6Impl : torch::nn::Module {
    ConvBNReLU    reduce_layer0{nullptr};   // c5 → c6
    BiFusionBlock Bifusion0{nullptr};       // (c4, c6) → c6 + extra c3 path
    RepBlock      Rep_p5_rb{nullptr};       BepC3 Rep_p5_bep{nullptr};
    ConvBNReLU    reduce_layer1{nullptr};   // c6 → c7
    BiFusionBlock Bifusion1{nullptr};
    RepBlock      Rep_p4_rb{nullptr};       BepC3 Rep_p4_bep{nullptr};
    ConvBNReLU    reduce_layer2{nullptr};   // c7 → c8
    BiFusionBlock Bifusion2{nullptr};
    RepBlock      Rep_p3_rb{nullptr};       BepC3 Rep_p3_bep{nullptr};
    ConvBNReLU    downsample2{nullptr};
    RepBlock      Rep_n4_rb{nullptr};       BepC3 Rep_n4_bep{nullptr};
    ConvBNReLU    downsample1{nullptr};
    RepBlock      Rep_n5_rb{nullptr};       BepC3 Rep_n5_bep{nullptr};
    ConvBNReLU    downsample0{nullptr};
    RepBlock      Rep_n6_rb{nullptr};       BepC3 Rep_n6_bep{nullptr};
    bool          use_rep_blocks = true;
    bool          use_repconv    = true;
    NeckP6Impl(int c2, int c3, int c4, int c5,
               int c6, int c7, int c8, int c9, int c10, int c11,
               int n_neck, bool use_rep_blocks, bool use_repconv,
               double bep_e);
  };

  std::shared_ptr<BackboneImpl>   backbone{nullptr};
  std::shared_ptr<NeckImpl>       neck{nullptr};
  std::shared_ptr<BackboneP6Impl> backbone_p6{nullptr};
  std::shared_ptr<NeckP6Impl>     neck_p6{nullptr};
  EffiDeHead                      detect{nullptr};
  bool                            is_p6 = false;

  // Old (nc, scale, reg_max, p6) ctor kept for backwards compat with
  // existing call sites; new (scale, nc) form added for trainer EMA
  // construction (matches v8/v11/v9/v3 convention).
  Yolo6Impl(int nc = 80, Yolo6Scale s = kYolo6s, int reg_max = 16,
            bool p6 = false);
  Yolo6Impl(Yolo6Scale s, int nc) : Yolo6Impl(nc, s, 16, false) {}

  // reg_max default for the Yolo6 trainer's V6DetectionLoss config.
  static constexpr int reg_max_default = 16;

  // Multi-scale raw feature maps for the loss (each [B, 4*reg_max+nc, H_i, W_i]
  // for DFL variants; [B, 4+nc, H_i, W_i] for n/s direct-branch). For now
  // wired for the DFL-headed m/l/m6/l6 variants — see V6DetectionLoss.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  // Returns 3 raw outputs in stride order (P3, P4, P5). Each is
  // [B, 4*(reg_max+1) + nc, H, W] — concatenated dist + cls per scale.
  std::vector<torch::Tensor> forward(torch::Tensor x);

  // Decoded inference output: [B, 4 + nc, A] xyxy in input pixels +
  // sigmoided cls — drop-in for our standard NMS.
  torch::Tensor forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo6);

}  // namespace yolocpp::models
