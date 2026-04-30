#include "yolocpp/models/yolo13.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolocpp::models {

namespace F = torch::nn::functional;

// External thread-local BN-eps switch (from yolo8.cpp).
double get_default_bn_eps();

// ───────── Scale helpers ─────────────────────────────────────────────────

int scale_channels_v13(int c, const Yolo13Scale& s) {
  c = std::min(c, s.max_channels);
  auto make_divisible = [](double v, int divisor) {
    return std::max(divisor, (int)std::round(v / divisor) * divisor);
  };
  return make_divisible(c * s.width_multiple, 8);
}

int scale_depth_v13(int n, const Yolo13Scale& s) {
  return std::max(1, (int)std::round(n * s.depth_multiple));
}

Yolo13Scale yolo13_scale_from_letter(const std::string& letter) {
  if (letter == "n") return kYolo13n;
  if (letter == "s") return kYolo13s;
  if (letter == "l") return kYolo13l;
  if (letter == "x") return kYolo13x;
  throw std::runtime_error("yolo13: unknown scale letter '" + letter +
                           "' (supported: n / s / l / x; iMoonLab does not "
                           "ship m)");
}

Yolo13Scale yolo13_scale_from_filename(const std::string& path) {
  // Look for "yolo13<letter>" anywhere in the path.
  std::string p = path;
  std::transform(p.begin(), p.end(), p.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  auto pos = p.find("yolo13");
  if (pos == std::string::npos) {
    throw std::runtime_error("yolo13_scale_from_filename: '" + path +
                             "' does not contain 'yolo13'");
  }
  if (pos + 6 >= p.size()) {
    throw std::runtime_error("yolo13_scale_from_filename: missing scale "
                             "letter after 'yolo13' in '" + path + "'");
  }
  std::string letter(1, p[pos + 6]);
  return yolo13_scale_from_letter(letter);
}

// ───────── DSConv ────────────────────────────────────────────────────────
//
// dw (groups=c_in, k×k, bias=False) → pw (1×1, bias=False) → BN → SiLU.

static int autopad_v13(int k, int p, int d = 1) {
  if (p >= 0) return p;
  (void)d;
  return k / 2;
}

DSConvImpl::DSConvImpl(int c_in, int c_out, int k, int s, int p, int d,
                       bool act)
    : act_silu(act) {
  int pad = autopad_v13(k, p, d);
  dw = register_module(
      "dw",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_in, k)
                            .stride(s)
                            .padding(pad)
                            .groups(c_in)
                            .dilation(d)
                            .bias(false)));
  pw = register_module(
      "pw",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 1)
                            .stride(1)
                            .padding(0)
                            .bias(false)));
  bn = register_module(
      "bn",
      torch::nn::BatchNorm2d(
          torch::nn::BatchNorm2dOptions(c_out).eps(get_default_bn_eps())));
}

torch::Tensor DSConvImpl::forward(torch::Tensor x) {
  x = pw(dw(x));
  x = bn(x);
  if (act_silu) x = F::silu(x);
  return x;
}

// ───────── DSBottleneck ──────────────────────────────────────────────────

DSBottleneckImpl::DSBottleneckImpl(int c1, int c2, bool shortcut, double e,
                                   int k1, int k2, int d2) {
  int c_ = (int)(c2 * e);
  cv1 = register_module(
      "cv1", DSConv(c1, c_, k1, /*s=*/1, /*p=*/-1, /*d=*/1, /*act=*/true));
  cv2 = register_module(
      "cv2", DSConv(c_, c2, k2, /*s=*/1, /*p=*/-1, /*d=*/d2, /*act=*/true));
  add = shortcut && (c1 == c2);
}

torch::Tensor DSBottleneckImpl::forward(torch::Tensor x) {
  auto y = cv2(cv1(x));
  return add ? x + y : y;
}

// ───────── DSC3k (C3-shaped, DSBottleneck inner) ─────────────────────────

DSC3kImpl::DSC3kImpl(int c1, int c2, int n, bool shortcut, double e, int k1,
                     int k2, int d2) {
  int c_ = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
  cv2 = register_module("cv2", Conv(c1, c_, 1, 1));
  cv3 = register_module("cv3", Conv(2 * c_, c2, 1));
  m   = register_module("m", torch::nn::Sequential());
  // Python: e=1.0 inside DSC3k's m loop, so DSBottleneck operates at full c_.
  for (int i = 0; i < n; ++i) {
    m->push_back(DSBottleneck(c_, c_, shortcut, /*e=*/1.0, k1, k2, d2));
  }
}

torch::Tensor DSC3kImpl::forward(torch::Tensor x) {
  auto a = cv1(x);
  // Sequential of DSBottleneck — torch's Sequential handles forward(Tensor).
  a = m->forward(a);
  auto b = cv2(x);
  return cv3(torch::cat({a, b}, /*dim=*/1));
}

// ───────── DSC3k2 (C2f-shaped, DSC3k or DSBottleneck inner) ─────────────

DSC3k2Impl::DSC3k2Impl(int c1, int c2, int n, bool dsc3k, double e,
                       bool shortcut, int k1, int k2, int d2) {
  c_inner = (int)(c2 * e);
  this->dsc3k = dsc3k;
  cv1 = register_module("cv1", Conv(c1, 2 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((2 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    if (dsc3k) {
      // Inner DSC3k uses n=2, e=1.0, k1=3, k2=k2 (from parent).
      m->push_back(
          DSC3k(c_inner, c_inner, /*n=*/2, shortcut, /*e=*/1.0, k1, k2, d2));
    } else {
      m->push_back(
          DSBottleneck(c_inner, c_inner, shortcut, /*e=*/1.0, k1, k2, d2));
    }
  }
}

torch::Tensor DSC3k2Impl::forward(torch::Tensor x) {
  auto y_split = cv1(x).chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> outs;
  outs.reserve(2 + m->size());
  outs.push_back(y_split[0]);
  outs.push_back(y_split[1]);
  for (size_t i = 0; i < m->size(); ++i) {
    torch::Tensor next;
    if (dsc3k) {
      auto* mod = m[i]->as<DSC3kImpl>();
      next = mod->forward(outs.back());
    } else {
      auto* mod = m[i]->as<DSBottleneckImpl>();
      next = mod->forward(outs.back());
    }
    outs.push_back(next);
  }
  return cv2(torch::cat(outs, /*dim=*/1));
}

// ───────── DownsampleConv ────────────────────────────────────────────────

DownsampleConvImpl::DownsampleConvImpl(int in_channels, bool channel_adjust_)
    : channel_adjust(channel_adjust_) {
  downsample = register_module(
      "downsample", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));
  if (channel_adjust) {
    channel_adjust_conv = register_module(
        "channel_adjust", Conv(in_channels, in_channels * 2, 1));
  }
  // When channel_adjust=False, Python uses nn.Identity() — we just skip.
}

torch::Tensor DownsampleConvImpl::forward(torch::Tensor x) {
  x = downsample(x);
  if (channel_adjust) x = channel_adjust_conv(x);
  return x;
}

// ───────── FullPAD_Tunnel ────────────────────────────────────────────────

FullPADTunnelImpl::FullPADTunnelImpl() {
  gate = register_parameter("gate", torch::zeros({}));
}

torch::Tensor FullPADTunnelImpl::forward(torch::Tensor a, torch::Tensor b) {
  return a + gate * b;
}

// ───────── FuseModule ────────────────────────────────────────────────────
//
// Python forward:
//   x1_ds = AvgPool2(x[0])            # /2 spatial
//   x3_up = NearestUpsample2(x[2])    # *2 spatial
//   cat   = concat(x1_ds, x[1], x3_up)  along channel
//   out   = conv_out(cat)
//
// channel_adjust=True  → conv_out: 4*c_in → c_in
// channel_adjust=False → conv_out: 3*c_in → c_in

FuseModuleImpl::FuseModuleImpl(int c_in, bool channel_adjust_)
    : channel_adjust(channel_adjust_) {
  downsample = register_module(
      "downsample", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));
  upsample = register_module(
      "upsample",
      torch::nn::Upsample(torch::nn::UpsampleOptions()
                              .scale_factor(std::vector<double>{2.0, 2.0})
                              .mode(torch::kNearest)));
  int cat_channels = channel_adjust ? 4 * c_in : 3 * c_in;
  conv_out = register_module("conv_out", Conv(cat_channels, c_in, 1));
}

torch::Tensor FuseModuleImpl::forward(std::vector<torch::Tensor> x) {
  // The forward is identical regardless of channel_adjust — both branches
  // cat exactly three tensors. `channel_adjust` only controls the in_channels
  // of conv_out at construction time. (parse_model decides: n/s scales use
  // channel_adjust=True because the cap-free L8 = 2 * L4 makes the cat
  // 4*c_in; l/x scales use channel_adjust=False because max_channels caps
  // L8 to L4, making the cat 3*c_in.)
  auto x1_ds = downsample(x[0]);
  auto x3_up = upsample(x[2]);
  return conv_out(torch::cat({x1_ds, x[1], x3_up}, /*dim=*/1));
}

// ───────── AdaHyperedgeGen ───────────────────────────────────────────────
//
// Implements iMoonLab's AdaHyperedgeGen verbatim. Eval-mode forward (the
// only path we use for inference) drops the dropout (identity) so this
// module is fully deterministic.
//
// Forward (B, N, D) → (B, N, M):
//   1. Pool tokens to context vector C ∈ R^{B × (D|2D)} (mean / max / both).
//   2. prototype_offsets = context_net(C).view(B, M, D).
//   3. prototypes        = prototype_base.unsqueeze(0) + prototype_offsets.
//   4. X_proj            = pre_head_proj(X).
//   5. Split X_proj into H heads → X_heads (B, H, N, head_dim).
//      Split prototypes into H heads → proto_heads (B, H, M, head_dim).
//   6. logits_h = bmm(X_heads, proto_heads^T) / scaling.
//   7. logits   = mean over heads (B, N, M).
//   8. (eval) softmax(logits, dim=N).
//
// State-dict keys (matches iMoonLab):
//   prototype_base       (M, D)              parameter
//   context_net.weight   (M*D, D|2D)         linear weight
//   context_net.bias     (M*D,)              linear bias
//   pre_head_proj.weight (D, D)              linear weight
//   pre_head_proj.bias   (D,)                linear bias

AdaHyperedgeGenImpl::AdaHyperedgeGenImpl(int node_dim, int num_hyperedges_,
                                        int num_heads_,
                                        const std::string& ctx)
    : num_heads(num_heads_),
      num_hyperedges(num_hyperedges_),
      head_dim(node_dim / num_heads_),
      context(ctx),
      scaling(std::sqrt((double)(node_dim / num_heads_))) {
  if (node_dim % num_heads_ != 0) {
    throw std::runtime_error(
        "AdaHyperedgeGen: node_dim must be divisible by num_heads");
  }
  if (ctx != "mean" && ctx != "max" && ctx != "both") {
    throw std::runtime_error(
        "AdaHyperedgeGen: context must be 'mean', 'max', or 'both' (got '" +
        ctx + "')");
  }
  prototype_base = register_parameter(
      "prototype_base", torch::empty({num_hyperedges, node_dim}));
  // Xavier-uniform init mirrors Python (overridden by .pt load).
  torch::nn::init::xavier_uniform_(prototype_base);

  int ctx_in = (ctx == "both") ? 2 * node_dim : node_dim;
  context_net = register_module(
      "context_net",
      torch::nn::Linear(torch::nn::LinearOptions(ctx_in,
                                                 num_hyperedges * node_dim)));
  pre_head_proj = register_module(
      "pre_head_proj",
      torch::nn::Linear(torch::nn::LinearOptions(node_dim, node_dim)));
}

torch::Tensor AdaHyperedgeGenImpl::forward(torch::Tensor X) {
  // X: (B, N, D)
  auto B = X.size(0);
  auto N = X.size(1);
  auto D = X.size(2);

  torch::Tensor context_cat;
  if (context == "mean") {
    context_cat = X.mean(/*dim=*/1);
  } else if (context == "max") {
    context_cat = std::get<0>(X.max(/*dim=*/1));
  } else {
    auto avg_ctx = X.mean(/*dim=*/1);
    auto max_ctx = std::get<0>(X.max(/*dim=*/1));
    context_cat = torch::cat({avg_ctx, max_ctx}, /*dim=*/-1);
  }
  // (B, M, D)
  auto prototype_offsets =
      context_net(context_cat).view({B, num_hyperedges, D});
  auto prototypes = prototype_base.unsqueeze(0) + prototype_offsets;

  // X_proj: (B, N, D) → (B, N, H, head_dim) → (B, H, N, head_dim)
  auto X_proj = pre_head_proj(X);
  auto X_heads = X_proj.view({B, N, num_heads, head_dim})
                     .transpose(1, 2)
                     .contiguous();
  // proto_heads: (B, M, D) → (B, M, H, head_dim) → (B, H, M, head_dim)
  auto proto_heads = prototypes.view({B, num_hyperedges, num_heads, head_dim})
                         .permute({0, 2, 1, 3})
                         .contiguous();

  // (BH, N, head_dim) and (BH, head_dim, M) → bmm → (BH, N, M)
  auto X_flat     = X_heads.reshape({B * num_heads, N, head_dim});
  auto proto_flat = proto_heads.reshape({B * num_heads, num_hyperedges, head_dim})
                       .transpose(1, 2);
  auto logits_h = torch::bmm(X_flat, proto_flat) / scaling;
  // (B, H, N, M) → mean over H → (B, N, M)
  auto logits =
      logits_h.view({B, num_heads, N, num_hyperedges}).mean(/*dim=*/1);

  // Dropout is identity in eval; we only use this module at inference.
  return torch::softmax(logits, /*dim=*/1);
}

// ───────── AdaHGConv ────────────────────────────────────────────────────
//
// Hypergraph convolution with vertex→edge→vertex message passing:
//   A   = edge_generator(X)            ∈ R^{B × N × M}
//   He  = bmm(Aᵀ, X)                   ∈ R^{B × M × D}      (vertex→edge)
//   He  = GELU(edge_proj.0(He))
//   X'  = bmm(A, He)                   ∈ R^{B × N × D}      (edge→vertex)
//   X'  = GELU(node_proj.0(X'))
//   out = X' + X                       (residual)
//
// Note edge_proj / node_proj are nn.Sequential of length 2: [Linear, GELU].
// Only Linear holds parameters — keys are <name>.0.{weight,bias}.

AdaHGConvImpl::AdaHGConvImpl(int embed_dim, int num_hyperedges,
                              int num_heads, const std::string& ctx) {
  edge_generator = register_module(
      "edge_generator",
      AdaHyperedgeGen(embed_dim, num_hyperedges, num_heads, ctx));

  // Sequential children registered as "0" (Linear) and "1" (GELU) so the
  // state_dict keys match Python's edge_proj.0.{weight,bias}.
  auto make_proj = [&](const std::string& name) {
    auto seq = torch::nn::Sequential();
    seq->push_back("0", torch::nn::Linear(
                            torch::nn::LinearOptions(embed_dim, embed_dim)));
    seq->push_back("1", torch::nn::GELU());
    return register_module(name, seq);
  };
  edge_proj = make_proj("edge_proj");
  node_proj = make_proj("node_proj");
}

torch::Tensor AdaHGConvImpl::forward(torch::Tensor X) {
  auto A  = edge_generator(X);                       // (B, N, M)
  auto He = torch::bmm(A.transpose(1, 2), X);        // (B, M, D)
  He      = edge_proj->forward(He);
  auto X_new = torch::bmm(A, He);                    // (B, N, D)
  X_new   = node_proj->forward(X_new);
  return X_new + X;
}

// ───────── AdaHGComputation ─────────────────────────────────────────────
//
// Thin 4D ↔ token wrapper around AdaHGConv:
//   tokens = x.flatten(2).transpose(1, 2)         (B, C, H, W) → (B, N, C)
//   tokens = hgnn(tokens)
//   x_out  = tokens.transpose(1, 2).view(B, C, H, W)

AdaHGComputationImpl::AdaHGComputationImpl(int dim, int num_hyperedges,
                                           int num_heads,
                                           const std::string& ctx)
    : embed_dim(dim) {
  hgnn = register_module(
      "hgnn", AdaHGConv(dim, num_hyperedges, num_heads, ctx));
}

torch::Tensor AdaHGComputationImpl::forward(torch::Tensor x) {
  auto B = x.size(0);
  auto C = x.size(1);
  auto H = x.size(2);
  auto W = x.size(3);
  auto tokens = x.flatten(/*start_dim=*/2).transpose(1, 2).contiguous();
  tokens = hgnn(tokens);
  return tokens.transpose(1, 2).contiguous().view({B, C, H, W});
}

// ───────── C3AH — CSP block with hypergraph branch ──────────────────────
//
// forward(x) = cv3( cat( m(cv1(x)), cv2(x) ) )
// num_heads in the hypergraph branch is c_/16 (asserted divisible).
// State-dict keys: cv1.{conv,bn}, cv2.{conv,bn}, cv3.{conv,bn},
//                  m.hgnn.<...>.

C3AHImpl::C3AHImpl(int c1, int c2, double e, int num_hyperedges,
                    const std::string& ctx) {
  int c_ = (int)(c2 * e);
  if (c_ % 16 != 0) {
    throw std::runtime_error(
        "C3AH: c_ = c2*e must be a multiple of 16 (got " +
        std::to_string(c_) + ")");
  }
  int num_heads = c_ / 16;
  cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
  cv2 = register_module("cv2", Conv(c1, c_, 1, 1));
  m   = register_module(
      "m", AdaHGComputation(c_, num_hyperedges, num_heads, ctx));
  cv3 = register_module("cv3", Conv(2 * c_, c2, 1));
}

torch::Tensor C3AHImpl::forward(torch::Tensor x) {
  return cv3(torch::cat({m(cv1(x)), cv2(x)}, /*dim=*/1));
}

// ───────── HyperACE — multi-scale hypergraph correlation enhancer ───────
//
// forward(X = [x_hi, x_mid, x_lo]):
//   x = fuse(X)
//   y = list(cv1(x).chunk(3, 1))         # split into 3 tensors of c_inner
//   out1 = branch1(y[1])
//   out2 = branch2(y[1])
//   for mi in m: y.append(mi(y[-1]))     # y grows by n
//   y[1] = out1
//   y.append(out2)
//   return cv2(cat(y, 1))                # cat → (4 + n) * c_inner
//
// Note that y[1] is OVERWRITTEN with out1 *after* the m-chain runs from
// y[-1] — but at the time the m-chain reads y[-1], if n=1 then y has
// length 3 (y[0], y[1] still original-cv1, y[2] original-cv1) → m[0] sees
// y[2] (the third chunk of cv1). So the substitution only affects the
// final cat.

HyperACEImpl::HyperACEImpl(int c1, int c2, int n, int num_hyperedges,
                            bool dsc3k, bool shortcut, double e1, double e2,
                            const std::string& ctx, bool channel_adjust) {
  c_inner = (int)(c2 * e1);
  cv1 = register_module("cv1", Conv(c1, 3 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((4 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    if (dsc3k) {
      m->push_back(
          DSC3k(c_inner, c_inner, /*n=*/2, shortcut, /*e=*/0.5,
                /*k1=*/3, /*k2=*/7));
    } else {
      m->push_back(DSBottleneck(c_inner, c_inner, shortcut));
    }
  }
  fuse = register_module("fuse", FuseModule(c1, channel_adjust));
  branch1 =
      register_module("branch1", C3AH(c_inner, c_inner, e2, num_hyperedges, ctx));
  branch2 =
      register_module("branch2", C3AH(c_inner, c_inner, e2, num_hyperedges, ctx));
}

torch::Tensor HyperACEImpl::forward(std::vector<torch::Tensor> X) {
  auto x = fuse->forward(X);
  auto chunks = cv1(x).chunk(3, /*dim=*/1);
  std::vector<torch::Tensor> y{chunks[0], chunks[1], chunks[2]};
  auto out1 = branch1->forward(y[1]);
  auto out2 = branch2->forward(y[1]);
  for (size_t i = 0; i < m->size(); ++i) {
    auto mi = m[i];
    torch::Tensor next;
    if (auto* d = mi->as<DSC3kImpl>()) {
      next = d->forward(y.back());
    } else if (auto* d = mi->as<DSBottleneckImpl>()) {
      next = d->forward(y.back());
    } else {
      throw std::runtime_error("HyperACE: unexpected inner module type");
    }
    y.push_back(next);
  }
  y[1] = out1;
  y.push_back(out2);
  return cv2(torch::cat(y, /*dim=*/1));
}

// ───────── V13AAttn ─────────────────────────────────────────────────────
//
// Forward (mirroring iMoonLab AAttn exactly):
//   qk_flat  = qk(x).flatten(2).transpose(1, 2)    (B, N, 2C)
//   v_4d     = v(x)                                (B, C, H, W)
//   pp       = pe(v_4d)                            (B, C, H, W)
//   v_flat   = v_4d.flatten(2).transpose(1, 2)     (B, N, C)
//   if area > 1: reshape qk_flat → (B*area, N/area, 2C); same for v_flat
//   q, k     = qk_flat.split([C, C], dim=2)
//   q,k,v    → reshape to (B, num_heads, head_dim, N) (per area)
//   attn     = softmax((qᵀ·k) * head_dim^-0.5, dim=-1)
//   x        = (v · attnᵀ).permute(0, 3, 1, 2)     # (B, N, H, head_dim)
//   if area > 1: undo windowing
//   x        = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
//   return proj(x + pp)

V13AAttnImpl::V13AAttnImpl(int dim, int num_heads_, int area_)
    : num_heads(num_heads_),
      head_dim(dim / num_heads_),
      area(area_) {
  int all_head_dim = head_dim * num_heads;
  qk   = register_module("qk",   Conv(dim, all_head_dim * 2, 1, 1,
                                       /*p=*/-1, /*g=*/1, /*act=*/false));
  v    = register_module("v",    Conv(dim, all_head_dim, 1, 1,
                                       /*p=*/-1, /*g=*/1, /*act=*/false));
  proj = register_module("proj", Conv(all_head_dim, dim, 1, 1,
                                       /*p=*/-1, /*g=*/1, /*act=*/false));
  // depthwise 5×5 pe with bias=False (v13 doesn't ship pe.conv.bias).
  pe   = register_module("pe",   Conv(all_head_dim, dim, /*k=*/5, /*s=*/1,
                                       /*p=*/2, /*g=*/dim, /*act=*/false));
}

torch::Tensor V13AAttnImpl::forward(torch::Tensor x) {
  auto B  = x.size(0);
  auto C  = x.size(1);
  auto H  = x.size(2);
  auto W  = x.size(3);
  auto N  = H * W;
  auto qk_flat = qk(x).flatten(2).transpose(1, 2).contiguous();   // (B, N, 2C)
  auto v_4d    = v(x);                                              // (B, C, H, W)
  auto pp      = pe(v_4d);                                          // (B, C, H, W)
  auto v_flat  = v_4d.flatten(2).transpose(1, 2).contiguous();      // (B, N, C)

  auto Bg = B;
  auto Ng = N;
  if (area > 1) {
    qk_flat = qk_flat.reshape({B * area, N / area, 2 * C});
    v_flat  = v_flat.reshape({B * area, N / area, C});
    Bg      = B * area;
    Ng      = N / area;
  }
  auto parts = qk_flat.split_with_sizes({C, C}, /*dim=*/2);
  auto q     = parts[0].transpose(1, 2).view({Bg, num_heads, head_dim, Ng});
  auto k     = parts[1].transpose(1, 2).view({Bg, num_heads, head_dim, Ng});
  auto v_h   = v_flat.transpose(1, 2).view({Bg, num_heads, head_dim, Ng});

  auto attn = (q.transpose(-2, -1).matmul(k))
                  .mul(1.0 / std::sqrt((double)head_dim));
  attn = attn.softmax(/*dim=*/-1);
  // (Bg, nh, hd, Ng) · (Bg, nh, Ng, Ng) → (Bg, nh, hd, Ng)
  auto out = v_h.matmul(attn.transpose(-2, -1));
  // permute to (Bg, Ng, nh, hd) then to spatial
  out = out.permute({0, 3, 1, 2}).contiguous();   // (Bg, Ng, nh, hd)

  if (area > 1) {
    out = out.reshape({B, N, C});
    Bg = B; Ng = N;
  } else {
    out = out.reshape({B, N, C});
  }
  out = out.reshape({B, H, W, C}).permute({0, 3, 1, 2}).contiguous();
  return proj(out + pp);
}

// ───────── V13ABlock ───────────────────────────────────────────────────

V13ABlockImpl::V13ABlockImpl(int dim, int num_heads, double mlp_ratio,
                              int area) {
  attn = register_module("attn", V13AAttn(dim, num_heads, area));
  int mlp_hidden = (int)(dim * mlp_ratio);
  auto seq = torch::nn::Sequential();
  seq->push_back("0", Conv(dim, mlp_hidden, 1, 1));
  seq->push_back("1", Conv(mlp_hidden, dim, 1, 1, /*p=*/-1, /*g=*/1,
                            /*act=*/false));
  mlp = register_module("mlp", seq);
}

torch::Tensor V13ABlockImpl::forward(torch::Tensor x) {
  x = x + attn->forward(x);
  x = x + mlp->forward(x);
  return x;
}

// ───────── V13A2C2f ────────────────────────────────────────────────────
//
// Same R-ELAN shape as v12 A2C2f but holds V13ABlock pairs (k=5 pe).

V13A2C2fImpl::V13A2C2fImpl(int c1, int c2, int n, bool a2_, int area,
                            bool residual_, double mlp_ratio, double e)
    : a2(a2_), residual(residual_) {
  c_inner = (int)(c2 * e);
  if (c_inner % 32 != 0) {
    throw std::runtime_error(
        "V13A2C2f: c_inner = c2*e must be multiple of 32 (got " +
        std::to_string(c_inner) + ")");
  }
  int nh = c_inner / 32;
  cv1 = register_module("cv1", Conv(c1, c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((1 + n) * c_inner, c2, 1, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    if (a2) {
      auto seq = torch::nn::Sequential();
      seq->push_back(V13ABlock(c_inner, nh, mlp_ratio, area));
      seq->push_back(V13ABlock(c_inner, nh, mlp_ratio, area));
      m->push_back(seq);
    } else {
      // a2=False path: Sequential of C3k. v13 yaml uses a2=True everywhere
      // for A2C2f, so this branch isn't actually reached for shipped models.
      m->push_back(C3k(c_inner, c_inner, /*n=*/2, /*shortcut=*/true,
                       /*g=*/1, /*e=*/0.5, /*k=*/3));
    }
  }
  if (a2 && residual && c1 == c2) {
    has_gamma = true;
    gamma = register_parameter("gamma", torch::full({c2}, 0.01));
  }
}

torch::Tensor V13A2C2fImpl::forward(torch::Tensor x) {
  std::vector<torch::Tensor> outs;
  outs.reserve(1 + m->size());
  outs.push_back(cv1(x));
  for (size_t i = 0; i < m->size(); ++i) {
    torch::Tensor next;
    if (a2) {
      next = m[i]->as<torch::nn::SequentialImpl>()->forward(outs.back());
    } else {
      next = m[i]->as<C3kImpl>()->forward(outs.back());
    }
    outs.push_back(next);
  }
  auto y = cv2(torch::cat(outs, /*dim=*/1));
  if (has_gamma) {
    auto g = gamma.view({1, -1, 1, 1});
    return x + g * y;
  }
  return y;
}

// ───────── Yolo13Detect (top-level) ──────────────────────────────────────
//
// 33-layer model with HyperACE + 5 FullPAD_Tunnels distributing the
// hypergraph-enhanced features across the neck. The constructor builds the
// modules at scaled channel counts; forward_train / forward_eval execute
// the per-layer schedule using a fixed `from`-table that mirrors the YAML.
//
// Per-scale `num_hyperedges` follows iMoonLab's parse_model: he = 8 * 0.5
// for n, 8 * 1.0 for s/l, 8 * 1.5 for x. channel_adjust=True for n/s,
// False for l/x.

namespace {

struct V13Step {
  std::vector<int> from;     // -1 = previous, otherwise absolute layer idx
  std::string      kind;
};

static const std::vector<V13Step> kV13Yaml = {
    /* 0  */ {{-1},      "Conv"},        // 64,3,2
    /* 1  */ {{-1},      "Conv"},        // 128,3,2,1,2
    /* 2  */ {{-1},      "DSC3k2"},      // 256,False,0.25
    /* 3  */ {{-1},      "Conv"},        // 256,3,2,1,4
    /* 4  */ {{-1},      "DSC3k2"},      // 512,False,0.25
    /* 5  */ {{-1},      "DSConv"},      // 512,3,2
    /* 6  */ {{-1},      "A2C2f"},       // 512,True,4
    /* 7  */ {{-1},      "DSConv"},      // 1024,3,2
    /* 8  */ {{-1},      "A2C2f"},       // 1024,True,1
    /* 9  */ {{4,6,8},   "HyperACE"},
    /* 10 */ {{-1},      "Up"},
    /* 11 */ {{9},       "DownsampleConv"},
    /* 12 */ {{6,9},     "FullPADTunnel"},
    /* 13 */ {{4,10},    "FullPADTunnel"},
    /* 14 */ {{8,11},    "FullPADTunnel"},
    /* 15 */ {{-1},      "Up"},
    /* 16 */ {{-1,12},   "Cat"},
    /* 17 */ {{-1},      "DSC3k2"},      // 512,True
    /* 18 */ {{-1,9},    "FullPADTunnel"},
    /* 19 */ {{17},      "Up"},
    /* 20 */ {{-1,13},   "Cat"},
    /* 21 */ {{-1},      "DSC3k2"},      // 256,True
    /* 22 */ {{10},      "Conv"},        // 256,1,1
    /* 23 */ {{21,22},   "FullPADTunnel"},
    /* 24 */ {{-1},      "Conv"},        // 256,3,2
    /* 25 */ {{-1,18},   "Cat"},
    /* 26 */ {{-1},      "DSC3k2"},      // 512,True
    /* 27 */ {{-1,9},    "FullPADTunnel"},
    /* 28 */ {{26},      "Conv"},        // 512,3,2
    /* 29 */ {{-1,14},   "Cat"},
    /* 30 */ {{-1},      "DSC3k2"},      // 1024,True
    /* 31 */ {{-1,11},   "FullPADTunnel"},
    /* 32 */ {{23,27,31},"Detect"},
};

}  // anonymous namespace

Yolo13DetectImpl::Yolo13DetectImpl(Yolo13Scale s, int nc_)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  // Per-scale num_hyperedges (parse_model rule).
  int he_base = 8;
  int he = he_base;
  if (s.depth_multiple == kYolo13n.depth_multiple &&
      s.width_multiple == kYolo13n.width_multiple) {
    he = (int)(he_base * 0.5);  // n
  } else if (s.depth_multiple == kYolo13x.depth_multiple &&
             s.width_multiple == kYolo13x.width_multiple) {
    he = (int)(he_base * 1.5);  // x
  }
  bool channel_adjust = !(s.depth_multiple == kYolo13l.depth_multiple &&
                          s.width_multiple == kYolo13l.width_multiple) &&
                        !(s.depth_multiple == kYolo13x.depth_multiple &&
                          s.width_multiple == kYolo13x.width_multiple);

  // Helper to scale channels.
  auto sc = [&](int c) { return scale_channels_v13(c, s); };
  auto sd = [&](int n) { return scale_depth_v13(n, s); };

  // parse_model: at scale l/x, every (D)SC3k2 gets c3k/dsc3k forced True
  // regardless of YAML.
  bool is_lx = (s.depth_multiple == kYolo13l.depth_multiple ||
                 s.depth_multiple == kYolo13x.depth_multiple) &&
                (s.width_multiple == kYolo13l.width_multiple ||
                 s.width_multiple == kYolo13x.width_multiple);

  // Track per-layer output channels for from-list resolution.
  std::vector<int> ch(kV13Yaml.size(), 0);

  // ── Backbone ──
  model->push_back(Conv(/*c_in=*/3,  sc(64),  /*k=*/3, /*s=*/2));               ch[0] = sc(64);
  model->push_back(Conv(ch[0],       sc(128), /*k=*/3, /*s=*/2, /*p=*/1, /*g=*/2)); ch[1] = sc(128);
  model->push_back(DSC3k2(ch[1], sc(256), sd(2),
                          /*dsc3k=*/(is_lx ? true : false), /*e=*/0.25,
                          /*shortcut=*/true));                                  ch[2] = sc(256);
  model->push_back(Conv(ch[2], sc(256), /*k=*/3, /*s=*/2, /*p=*/1, /*g=*/4));   ch[3] = sc(256);
  model->push_back(DSC3k2(ch[3], sc(512), sd(2),
                          /*dsc3k=*/(is_lx ? true : false), /*e=*/0.25,
                          /*shortcut=*/true));                                  ch[4] = sc(512);
  // parse_model: at scale l/x, A2C2f gets residual=True and mlp_ratio=1.5
  // appended (as the C3k2/DSC3k2 c3k=True override).
  bool a2c2f_residual = (s.depth_multiple == kYolo13l.depth_multiple ||
                         s.depth_multiple == kYolo13x.depth_multiple) &&
                        (s.width_multiple == kYolo13l.width_multiple ||
                         s.width_multiple == kYolo13x.width_multiple);
  double a2c2f_mlp = a2c2f_residual ? 1.5 : 2.0;
  model->push_back(DSConv(ch[4], sc(512), /*k=*/3, /*s=*/2));                   ch[5] = sc(512);
  model->push_back(V13A2C2f(ch[5], sc(512), sd(4), /*a2=*/true, /*area=*/4,
                              a2c2f_residual, a2c2f_mlp));                       ch[6] = sc(512);
  model->push_back(DSConv(ch[6], sc(1024), /*k=*/3, /*s=*/2));                  ch[7] = sc(1024);
  model->push_back(V13A2C2f(ch[7], sc(1024), sd(4), /*a2=*/true, /*area=*/1,
                              a2c2f_residual, a2c2f_mlp));                       ch[8] = sc(1024);

  // ── L9: HyperACE (parse_model: c1 = ch[f[1]] = ch[6]; n = scaled yaml n) ─
  int c2_l9 = sc(512);
  int hyper_n = sd(2);  // yaml n=2, depth-scaled
  model->push_back(HyperACE(/*c1=*/ch[6], /*c2=*/c2_l9, /*n=*/hyper_n,
                             /*he=*/he,
                             /*dsc3k=*/true, /*shortcut=*/true,
                             /*e1=*/0.5, /*e2=*/1.0, /*ctx=*/"both",
                             /*channel_adjust=*/channel_adjust));
  ch[9] = c2_l9;

  // L10: Upsample (Identity wrapper — operate dynamically in forward).
  // We use a placeholder Conv with k=1, in=out=ch[9] and copy in forward...
  // Cleaner: use torch::nn::Upsample directly (no params).
  model->push_back(torch::nn::Upsample(
      torch::nn::UpsampleOptions()
          .scale_factor(std::vector<double>{2.0, 2.0})
          .mode(torch::kNearest)));
  ch[10] = ch[9];

  // L11: DownsampleConv. parse_model passes channel_adjust=False for l/x
  // (and clamps c2 = c1); n/s use default True (c2 = 2*c1).
  bool ds_channel_adjust = channel_adjust;  // same gate as HyperACE.fuse
  model->push_back(DownsampleConv(ch[9], ds_channel_adjust));
  ch[11] = ds_channel_adjust ? ch[9] * 2 : ch[9];

  // L12-14, 18, 23, 27, 31: FullPAD_Tunnels (no channel change).
  model->push_back(FullPADTunnel());  ch[12] = ch[6];
  model->push_back(FullPADTunnel());  ch[13] = ch[4];
  model->push_back(FullPADTunnel());  ch[14] = ch[8];

  // L15: Upsample.
  model->push_back(torch::nn::Upsample(
      torch::nn::UpsampleOptions()
          .scale_factor(std::vector<double>{2.0, 2.0})
          .mode(torch::kNearest)));
  ch[15] = ch[14];

  // L16: Concat[-1, 12]
  ch[16] = ch[15] + ch[12];
  // Concat is structural (no module) — we use a torch::nn::Identity slot
  // so positional iteration over `model` lines up with the YAML.
  model->push_back(torch::nn::Identity());

  // L17: DSC3k2 [512, True] dsc3k=True
  model->push_back(DSC3k2(ch[16], sc(512), sd(2), /*dsc3k=*/true, /*e=*/0.5,
                           /*shortcut=*/true));
  ch[17] = sc(512);

  // L18: FullPAD_Tunnel
  model->push_back(FullPADTunnel());  ch[18] = ch[17];

  // L19: Upsample(17)
  model->push_back(torch::nn::Upsample(
      torch::nn::UpsampleOptions()
          .scale_factor(std::vector<double>{2.0, 2.0})
          .mode(torch::kNearest)));
  ch[19] = ch[17];

  // L20: Concat[-1, 13]
  ch[20] = ch[19] + ch[13];
  model->push_back(torch::nn::Identity());

  // L21: DSC3k2 [256, True] dsc3k=True
  model->push_back(DSC3k2(ch[20], sc(256), sd(2), /*dsc3k=*/true, /*e=*/0.5,
                           /*shortcut=*/true));
  ch[21] = sc(256);

  // L22: Conv 1×1 from L10
  model->push_back(Conv(ch[10], sc(256), /*k=*/1, /*s=*/1));
  ch[22] = sc(256);

  // L23: FullPAD_Tunnel
  model->push_back(FullPADTunnel());  ch[23] = ch[21];

  // L24: Conv 3×3 s=2 from L23
  model->push_back(Conv(ch[23], sc(256), /*k=*/3, /*s=*/2));
  ch[24] = sc(256);

  // L25: Concat[-1, 18]
  ch[25] = ch[24] + ch[18];
  model->push_back(torch::nn::Identity());

  // L26: DSC3k2 [512, True] dsc3k=True
  model->push_back(DSC3k2(ch[25], sc(512), sd(2), /*dsc3k=*/true, /*e=*/0.5,
                           /*shortcut=*/true));
  ch[26] = sc(512);

  // L27: FullPAD_Tunnel
  model->push_back(FullPADTunnel());  ch[27] = ch[26];

  // L28: Conv 3×3 s=2 from L26
  model->push_back(Conv(ch[26], sc(512), /*k=*/3, /*s=*/2));
  ch[28] = sc(512);

  // L29: Concat[-1, 14]
  ch[29] = ch[28] + ch[14];
  model->push_back(torch::nn::Identity());

  // L30: DSC3k2 [1024, True] dsc3k=True
  model->push_back(DSC3k2(ch[29], sc(1024), sd(2), /*dsc3k=*/true, /*e=*/0.5,
                           /*shortcut=*/true));
  ch[30] = sc(1024);

  // L31: FullPAD_Tunnel
  model->push_back(FullPADTunnel());  ch[31] = ch[30];

  // L32: Detect from [23, 27, 31], legacy=false
  std::vector<int> det_ch = {ch[23], ch[27], ch[31]};
  model->push_back(Detect(nc, det_ch, /*legacy=*/false));

  // Probe strides via a one-shot dummy forward — same as v8 / v11.
  stride = {8.0, 16.0, 32.0};  // P3/P4/P5
  auto* d = model[32]->as<DetectImpl>();
  d->stride = stride;
}

namespace {

// Resolve a from-index into the per-layer output cache. -1 means
// "previous output" — but in our case `prev` is always outs[i-1].
inline torch::Tensor resolve_from(int f, int i,
                                   const std::vector<torch::Tensor>& outs,
                                   const torch::Tensor& prev) {
  if (f == -1) return prev;
  return outs[f];
}

}  // namespace

std::vector<torch::Tensor> Yolo13DetectImpl::forward_train(torch::Tensor x) {
  std::vector<torch::Tensor> outs(kV13Yaml.size());
  torch::Tensor prev = x;
  for (size_t i = 0; i < kV13Yaml.size(); ++i) {
    const auto& step = kV13Yaml[i];
    auto module = model[i];
    torch::Tensor y;
    if (step.kind == "Conv") {
      y = module->as<ConvImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DSConv") {
      y = module->as<DSConvImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DSC3k2") {
      y = module->as<DSC3k2Impl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "A2C2f") {
      y = module->as<V13A2C2fImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "Up") {
      y = module->as<torch::nn::UpsampleImpl>()->forward(
          resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DownsampleConv") {
      y = module->as<DownsampleConvImpl>()->forward(
          resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "HyperACE") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from) ins.push_back(outs[f]);
      y = module->as<HyperACEImpl>()->forward(ins);
    } else if (step.kind == "FullPADTunnel") {
      auto a = resolve_from(step.from[0], i, outs, prev);
      auto b = resolve_from(step.from[1], i, outs, prev);
      y = module->as<FullPADTunnelImpl>()->forward(a, b);
    } else if (step.kind == "Cat") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from)
        ins.push_back(resolve_from(f, i, outs, prev));
      y = torch::cat(ins, /*dim=*/1);
    } else if (step.kind == "Detect") {
      std::vector<torch::Tensor> feats;
      for (int f : step.from) feats.push_back(outs[f]);
      auto* det = module->as<DetectImpl>();
      auto out_levels = det->forward_features(feats);
      outs[i] = out_levels.front();  // not used; placeholder
      return out_levels;
    } else {
      throw std::runtime_error("Yolo13Detect: unknown step kind '" +
                                step.kind + "'");
    }
    outs[i] = y;
    prev = y;
  }
  // Detect returns above; reaching here means we never hit Detect.
  throw std::runtime_error("Yolo13Detect::forward_train: missing Detect");
}

torch::Tensor Yolo13DetectImpl::forward_eval(torch::Tensor x) {
  std::vector<torch::Tensor> outs(kV13Yaml.size());
  torch::Tensor prev = x;
  for (size_t i = 0; i < kV13Yaml.size(); ++i) {
    const auto& step = kV13Yaml[i];
    auto module = model[i];
    torch::Tensor y;
    if (step.kind == "Conv") {
      y = module->as<ConvImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DSConv") {
      y = module->as<DSConvImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DSC3k2") {
      y = module->as<DSC3k2Impl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "A2C2f") {
      y = module->as<V13A2C2fImpl>()->forward(resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "Up") {
      y = module->as<torch::nn::UpsampleImpl>()->forward(
          resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "DownsampleConv") {
      y = module->as<DownsampleConvImpl>()->forward(
          resolve_from(step.from[0], i, outs, prev));
    } else if (step.kind == "HyperACE") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from) ins.push_back(outs[f]);
      y = module->as<HyperACEImpl>()->forward(ins);
    } else if (step.kind == "FullPADTunnel") {
      auto a = resolve_from(step.from[0], i, outs, prev);
      auto b = resolve_from(step.from[1], i, outs, prev);
      y = module->as<FullPADTunnelImpl>()->forward(a, b);
    } else if (step.kind == "Cat") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from)
        ins.push_back(resolve_from(f, i, outs, prev));
      y = torch::cat(ins, /*dim=*/1);
    } else if (step.kind == "Detect") {
      std::vector<torch::Tensor> feats;
      for (int f : step.from) feats.push_back(outs[f]);
      auto* det = module->as<DetectImpl>();
      auto raw = det->forward_features(feats);
      return det->decode(raw);
    } else {
      throw std::runtime_error("Yolo13Detect: unknown step kind '" +
                                step.kind + "'");
    }
    outs[i] = y;
    prev = y;
  }
  throw std::runtime_error("Yolo13Detect::forward_eval: missing Detect");
}

int Yolo13DetectImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  // Reuse a generic name-based loader: copy each <key>=<tensor> into
  // named_parameters / named_buffers when the shape matches.
  torch::NoGradGuard ng;
  std::unordered_map<std::string, at::Tensor> by_name;
  by_name.reserve(entries.size());
  for (auto& [k, t] : entries) by_name[k] = t;

  int copied = 0;
  auto try_load = [&](const std::string& k, torch::Tensor t) {
    auto it = by_name.find(k);
    if (it == by_name.end()) return;
    if (!t.sizes().equals(it->second.sizes())) {
      throw std::runtime_error(
          "yolo13 load: shape mismatch for '" + k + "': ours=" +
          std::string(c10::str(t.sizes())) + " ckpt=" +
          std::string(c10::str(it->second.sizes())));
    }
    t.copy_(it->second.to(t.dtype()));
    ++copied;
  };
  for (auto& p : this->named_parameters()) try_load(p.key(), p.value());
  for (auto& p : this->named_buffers())    try_load(p.key(), p.value());
  return copied;
}

}  // namespace yolocpp::models
