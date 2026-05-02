#include "yolocpp/models/yolo10.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolocpp::models {

namespace {
inline int md8(int v) { return ((v + 7) / 8) * 8; }
}  // namespace

// ─── SCDown ──────────────────────────────────────────────────────────────
SCDownImpl::SCDownImpl(int c1, int c2, int k, int s) {
  cv1 = register_module("cv1", Conv(c1, c2, 1, 1));
  cv2 = register_module("cv2", DWConv(c2, c2, /*k=*/k, /*s=*/s, /*act=*/false));
}
torch::Tensor SCDownImpl::forward(torch::Tensor x) {
  return cv2(cv1(x));
}

// ─── RepVGGDW (deploy form) — single 7×7 DWConv + SiLU ───────────────────
RepVGGDWImpl::RepVGGDWImpl(int c) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c, c, 7)
                            .stride(1).padding(3).groups(c).bias(true)));
}
torch::Tensor RepVGGDWImpl::forward(torch::Tensor x) {
  return torch::silu(conv(x));
}

// ─── CIB ─────────────────────────────────────────────────────────────────
CIBImpl::CIBImpl(int c1, int c2, bool shortcut, bool lk_, double e) {
  add = shortcut && (c1 == c2);
  lk  = lk_;
  const int c_ = static_cast<int>(c2 * e);
  cv1 = register_module("cv1", torch::nn::ModuleList());
  cv1->push_back(Conv(c1,    c1,    /*k=*/3, /*s=*/1, /*p=*/-1, /*g=*/c1));      // 0: DWConv 3×3
  cv1->push_back(Conv(c1,    2 * c_, /*k=*/1, /*s=*/1));                         // 1: pointwise 1×1
  if (lk) {
    cv1->push_back(RepVGGDW(2 * c_));                                            // 2: RepVGGDW
  } else {
    cv1->push_back(Conv(2 * c_, 2 * c_, /*k=*/3, /*s=*/1, /*p=*/-1, /*g=*/2 * c_));  // 2: DWConv 3×3
  }
  cv1->push_back(Conv(2 * c_, c2,     /*k=*/1, /*s=*/1));                        // 3: pointwise 1×1
  cv1->push_back(Conv(c2,    c2,    /*k=*/3, /*s=*/1, /*p=*/-1, /*g=*/c2));      // 4: DWConv 3×3
}
torch::Tensor CIBImpl::forward(torch::Tensor x) {
  auto y = x;
  y = cv1[0]->as<ConvImpl>()->forward(y);
  y = cv1[1]->as<ConvImpl>()->forward(y);
  if (lk) y = cv1[2]->as<RepVGGDWImpl>()->forward(y);
  else    y = cv1[2]->as<ConvImpl>()->forward(y);
  y = cv1[3]->as<ConvImpl>()->forward(y);
  y = cv1[4]->as<ConvImpl>()->forward(y);
  return add ? (x + y) : y;
}

// ─── C2fCIB ──────────────────────────────────────────────────────────────
C2fCIBImpl::C2fCIBImpl(int c1, int c2, int n, bool shortcut, bool lk) {
  c_inner = c2 / 2;
  cv1 = register_module("cv1", Conv(c1, 2 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((2 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  // Upstream: `nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk))`.
  // The e=1.0 makes the middle RepVGGDW operate on `2*c_inner` channels.
  for (int i = 0; i < n; ++i)
    m->push_back(CIB(c_inner, c_inner, shortcut, lk, /*e=*/1.0));
}
torch::Tensor C2fCIBImpl::forward(torch::Tensor x) {
  auto y      = cv1(x);
  auto chunks = y.chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> outs = {chunks[0], chunks[1]};
  for (size_t i = 0; i < m->size(); ++i) {
    outs.push_back(m[i]->as<CIBImpl>()->forward(outs.back()));
  }
  return cv2(torch::cat(outs, /*dim=*/1));
}

// ─── PSA ─────────────────────────────────────────────────────────────────
PSAImpl::PSAImpl(int c1, int c2) {
  c_  = c1 / 2;
  cv1 = register_module("cv1", Conv(c1, 2 * c_, 1, 1));
  cv2 = register_module("cv2", Conv(2 * c_, c2, 1, 1));
  attn = register_module("attn",
                         PSAAttention(c_, /*num_heads=*/std::max(1, c_ / 64),
                                       /*attn_ratio=*/0.5));
  ffn  = register_module("ffn",
                         torch::nn::Sequential(Conv(c_, c_ * 2, 1),
                                                Conv(c_ * 2, c_, 1, 1, -1, 1, /*act=*/false)));
}
torch::Tensor PSAImpl::forward(torch::Tensor x) {
  auto y      = cv1(x);
  auto chunks = y.chunk(2, /*dim=*/1);
  auto a = chunks[0];
  auto b = chunks[1];
  b = b + attn->forward(b);
  b = b + ffn->forward(b);
  return cv2(torch::cat({a, b}, /*dim=*/1));
}

// ─── yolov10 yaml-walker (scale-parametric) ──────────────────────────────
namespace {
struct Spec {
  std::vector<int> from;
  int              n_repeat;       // post-scale (depth already applied)
  std::string      kind;
  std::vector<int> args;           // first arg = c2 (post-scale), others
                                    // passthrough; may include
                                    // {shortcut(0/1), lk(0/1)} flags
};

// Apply the upstream parse_model channel scaling:
//   c2 = make_divisible(min(c2_yaml, max_channels) * width, 8)
inline int v10_scale_ch(int c2_yaml, const Yolo10Scale& s) {
  int capped = std::min(c2_yaml, s.max_channels);
  int scaled = (int)std::round(capped * s.width);
  return ((scaled + 7) / 8) * 8;
}
inline int v10_scale_n(int n_yaml, const Yolo10Scale& s) {
  return std::max(1, (int)std::round(n_yaml * s.depth));
}

// Generate the per-scale Spec list. Topology is fixed; what differs by
// scale is the kind at layers 6, 8, 13, 19 (C2f vs C2fCIB) and the
// shortcut/lk flags on the C2fCIBs.
std::vector<Spec> build_v10_yaml(const Yolo10Scale& s) {
  // Helpers — n is pre-scale (yaml repeats column).
  auto C  = [](std::vector<int> from, std::string kind, std::vector<int> args, int n = 1) {
    return Spec{std::move(from), n, std::move(kind), std::move(args)};
  };

  // Width comparisons against the canonical scale constants. We can't use
  // operator== on a struct without it, so compare fields explicitly.
  const bool is_n = (s.depth == 0.33 && s.width == 0.25);
  const bool is_s = (s.depth == 0.33 && s.width == 0.50);
  const bool is_m = (s.depth == 0.67 && s.width == 0.75);
  const bool is_b = (s.depth == 0.67 && s.width == 1.00);
  const bool is_l = (s.depth == 1.00 && s.width == 1.00);
  const bool is_x = (s.depth == 1.00 && s.width == 1.25);
  (void)is_l; (void)is_b;

  // Per-position kind decisions, mirroring upstream yolov10{n,s,m,b,l,x}.yaml.
  // Layer 6 (P4 backbone): only `x` uses C2fCIB.
  const bool l6_cib  = is_x;
  // Layer 8 (P5 backbone): n keeps C2f; s/m/b/l/x use C2fCIB. lk=true only on s.
  const bool l8_cib  = !is_n;
  const bool l8_lk   = is_s;
  // Layer 13 (P4 head): n/s/m use C2f; b/l/x use C2fCIB.
  const bool l13_cib = !(is_n || is_s || is_m);
  // Layer 19 (P4 head out): n/s use C2f; m/b/l/x use C2fCIB.
  const bool l19_cib = !(is_n || is_s);
  // Layer 22 (P5 head): always C2fCIB. lk=true on n/s only.
  const bool l22_lk  = (is_n || is_s);

  // Helper to emit a C2f-or-C2fCIB layer with the right channel.
  auto C2fOrCIB = [&](std::vector<int> from, int c2_yaml, int n_yaml,
                      bool cib, bool shortcut, bool lk) -> Spec {
    int c2 = v10_scale_ch(c2_yaml, s);
    int n  = v10_scale_n(n_yaml, s);
    if (cib) {
      return C(std::move(from), "C2fCIB",
               {c2, shortcut ? 1 : 0, lk ? 1 : 0}, n);
    }
    return C(std::move(from), "C2f",
             {c2, shortcut ? 1 : 0}, n);
  };

  std::vector<Spec> y;
  // Backbone
  y.push_back(C({-1}, "Conv",   {v10_scale_ch(64,   s), 3, 2}));                  // 0  P1/2
  y.push_back(C({-1}, "Conv",   {v10_scale_ch(128,  s), 3, 2}));                  // 1  P2/4
  y.push_back(C2fOrCIB({-1}, 128, 3, /*cib=*/false, /*sc=*/true, /*lk=*/false));  // 2
  y.push_back(C({-1}, "Conv",   {v10_scale_ch(256,  s), 3, 2}));                  // 3  P3/8
  y.push_back(C2fOrCIB({-1}, 256, 6, /*cib=*/false, /*sc=*/true, /*lk=*/false));  // 4
  y.push_back(C({-1}, "SCDown", {v10_scale_ch(512,  s), 3, 2}));                  // 5  P4/16
  y.push_back(C2fOrCIB({-1}, 512, 6, l6_cib,        /*sc=*/true, /*lk=*/false));  // 6
  y.push_back(C({-1}, "SCDown", {v10_scale_ch(1024, s), 3, 2}));                  // 7  P5/32
  y.push_back(C2fOrCIB({-1}, 1024, 3, l8_cib,       /*sc=*/true, l8_lk));         // 8
  y.push_back(C({-1}, "SPPF",   {v10_scale_ch(1024, s), 5}));                     // 9
  y.push_back(C({-1}, "PSA",    {v10_scale_ch(1024, s)}));                        // 10
  // Head — top-down
  y.push_back(C({-1},     "Upsample", {2}));                                      // 11
  y.push_back(C({-1, 6},  "Concat",   {}));                                       // 12
  y.push_back(C2fOrCIB({-1}, 512, 3, l13_cib, /*sc=*/l13_cib, /*lk=*/false));     // 13
  y.push_back(C({-1},     "Upsample", {2}));                                      // 14
  y.push_back(C({-1, 4},  "Concat",   {}));                                       // 15
  y.push_back(C2fOrCIB({-1}, 256, 3, /*cib=*/false, /*sc=*/false, /*lk=*/false)); // 16  P3 head
  // Head — bottom-up
  y.push_back(C({-1}, "Conv",         {v10_scale_ch(256, s), 3, 2}));             // 17
  y.push_back(C({-1, 13}, "Concat",   {}));                                       // 18
  y.push_back(C2fOrCIB({-1}, 512, 3, l19_cib, /*sc=*/l19_cib, /*lk=*/false));     // 19  P4 head
  y.push_back(C({-1}, "SCDown",       {v10_scale_ch(512, s), 3, 2}));             // 20
  y.push_back(C({-1, 10}, "Concat",   {}));                                       // 21
  y.push_back(C2fOrCIB({-1}, 1024, 3, /*cib=*/true, /*sc=*/true, l22_lk));        // 22  P5 head
  y.push_back(C({16, 19, 22}, "Detect", {}));                                     // 23
  return y;
}

const std::vector<Spec>& v10_yaml_for(const Yolo10Scale& s) {
  // Cache one yaml per scale instance encountered. The scale set is small
  // (6 canonical) and identity-by-fields is enough.
  static std::vector<std::pair<Yolo10Scale, std::vector<Spec>>> cache;
  for (auto& [k, v] : cache) {
    if (k.depth == s.depth && k.width == s.width &&
        k.max_channels == s.max_channels) {
      return v;
    }
  }
  cache.push_back({s, build_v10_yaml(s)});
  return cache.back().second;
}
}  // namespace

Yolo10Impl::Yolo10Impl(Yolo10Scale s, int nc_)
    : Yolo10Impl(s, nc_, /*dual_head=*/false) {}

Yolo10Impl::Yolo10Impl(Yolo10Scale scale_, int nc_, bool dual_head_)
    : scale(scale_), nc(nc_), dual_head(dual_head_) {
  model = register_module("model", torch::nn::ModuleList());
  const auto& yaml = v10_yaml_for(scale);
  std::vector<int> ch;
  const int c_in_img = 3;

  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };
  auto in_ch_for = [&](size_t i) -> int {
    const auto& s = yaml[i];
    if (s.kind == "Concat") {
      int sum = 0;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        sum += (idx == -1) ? c_in_img : ch[idx];
      }
      return sum;
    }
    int f = s.from[0];
    int idx = resolve_idx(f, (int)i);
    return (idx == -1) ? c_in_img : ch[idx];
  };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    int in_ch = in_ch_for(i);
    // args[0] is already post-scale (md8 applied in v10_scale_ch); pass it
    // through verbatim. The legacy md8(args[0]) wrap was needed when only n
    // was wired and channels were hand-scaled in the yaml.
    int c2 = s.args.empty() ? 0 : s.args[0];

    if (s.kind == "Conv") {
      model->push_back(Conv(in_ch, c2, s.args[1], s.args[2]));
      ch.push_back(c2);
    } else if (s.kind == "C2f") {
      bool shortcut = s.args.size() > 1 && s.args[1] != 0;
      model->push_back(C2f(in_ch, c2, s.n_repeat, shortcut));
      ch.push_back(c2);
    } else if (s.kind == "C2fCIB") {
      bool shortcut = s.args.size() > 1 && s.args[1] != 0;
      bool lk       = s.args.size() > 2 && s.args[2] != 0;
      model->push_back(C2fCIB(in_ch, c2, s.n_repeat, shortcut, lk));
      ch.push_back(c2);
    } else if (s.kind == "SCDown") {
      model->push_back(SCDown(in_ch, c2, s.args[1], s.args[2]));
      ch.push_back(c2);
    } else if (s.kind == "SPPF") {
      model->push_back(SPPF(in_ch, c2, /*k=*/s.args[1]));
      ch.push_back(c2);
    } else if (s.kind == "PSA") {
      model->push_back(PSA(in_ch, c2));
      ch.push_back(c2);
    } else if (s.kind == "Upsample") {
      double sf = (double)s.args[0];
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{sf, sf})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (s.kind == "Concat") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    } else if (s.kind == "Detect") {
      std::vector<int> det_ch;
      for (int f : s.from) det_ch.push_back(ch[f]);
      // v10's one2one head uses the v11-style cv3 (DWConvBlock×2 + Conv2d).
      model->push_back(Detect(nc, det_ch, /*legacy=*/false));
      ch.push_back(0);
      // Optional one2many head (paper §3.1) — v8-style cv3 (Conv-Conv-Conv2d,
      // legacy=true). Lives outside the `model` ModuleList so its
      // state-dict keys don't collide with the one2one head's cv2/cv3.
      if (dual_head) {
        o2m_detect = register_module("o2m_detect",
                                      Detect(nc, det_ch, /*legacy=*/true));
      }
    } else {
      throw std::runtime_error("yolo10: unknown layer kind '" + s.kind + "'");
    }
  }
}

torch::Tensor Yolo10Impl::forward_eval(torch::Tensor x) {
  const auto& yaml = v10_yaml_for(scale);
  std::vector<torch::Tensor> outs(yaml.size());
  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        parts.push_back(idx == -1 ? x : outs[idx]);
      }
      in = torch::cat(parts, /*dim=*/1);
    } else if (s.kind != "Detect") {
      int f   = s.from[0];
      int idx = resolve_idx(f, (int)i);
      in      = (idx == -1) ? x : outs[idx];
    }

    if      (s.kind == "Conv")     outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "C2f")      outs[i] = model[i]->as<C2fImpl>()->forward(in);
    else if (s.kind == "C2fCIB")   outs[i] = model[i]->as<C2fCIBImpl>()->forward(in);
    else if (s.kind == "SCDown")   outs[i] = model[i]->as<SCDownImpl>()->forward(in);
    else if (s.kind == "SPPF")     outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (s.kind == "PSA")      outs[i] = model[i]->as<PSAImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
    else if (s.kind == "Detect") {
      auto* d = model[i]->as<DetectImpl>();
      std::vector<torch::Tensor> det_in;
      for (int f : s.from) det_in.push_back(outs[f]);
      if (stride.empty()) {
        int img_h = (int)x.size(2);
        for (auto& t : det_in) stride.push_back((double)img_h / (double)t.size(2));
        d->stride = stride;
      }
      auto feats = d->forward_features(det_in);
      outs[i]    = d->decode(feats);
    }
  }
  return outs.back();
}

std::vector<torch::Tensor> Yolo10Impl::forward_train(torch::Tensor x) {
  // Mirror of forward_eval up to (but not through) the Detect decode.
  // Returns per-scale raw [B, 4*reg_max+nc, H_i, W_i] feature maps:
  //   dual_head=false: 3 tensors (one2one P3, P4, P5).
  //   dual_head=true : 6 tensors {o2m P3..P5, o2o P3..P5} for V10DualLoss
  //                    (consistent assignment per paper §3.1).
  const auto& yaml = v10_yaml_for(scale);
  std::vector<torch::Tensor> outs(yaml.size());
  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        parts.push_back(idx == -1 ? x : outs[idx]);
      }
      in = torch::cat(parts, /*dim=*/1);
    } else if (s.kind != "Detect") {
      int f   = s.from[0];
      int idx = resolve_idx(f, (int)i);
      in      = (idx == -1) ? x : outs[idx];
    }

    if      (s.kind == "Conv")     outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "C2f")      outs[i] = model[i]->as<C2fImpl>()->forward(in);
    else if (s.kind == "C2fCIB")   outs[i] = model[i]->as<C2fCIBImpl>()->forward(in);
    else if (s.kind == "SCDown")   outs[i] = model[i]->as<SCDownImpl>()->forward(in);
    else if (s.kind == "SPPF")     outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (s.kind == "PSA")      outs[i] = model[i]->as<PSAImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
    else if (s.kind == "Detect") {
      auto* d = model[i]->as<DetectImpl>();
      std::vector<torch::Tensor> det_in;
      for (int f : s.from) det_in.push_back(outs[f]);
      if (stride.empty()) {
        int img_h = (int)x.size(2);
        for (auto& t : det_in) stride.push_back((double)img_h / (double)t.size(2));
        d->stride = stride;
        if (dual_head && o2m_detect) o2m_detect->stride = stride;
      }
      auto o2o_feats = d->forward_features(det_in);
      if (!dual_head) return o2o_feats;
      auto o2m_feats = o2m_detect->forward_features(det_in);
      // [o2m P3..P5, o2o P3..P5]
      std::vector<torch::Tensor> joined;
      joined.reserve(o2m_feats.size() + o2o_feats.size());
      for (auto& t : o2m_feats) joined.push_back(std::move(t));
      for (auto& t : o2o_feats) joined.push_back(std::move(t));
      return joined;
    }
  }
  TORCH_CHECK(false, "Yolo10Impl::forward_train: no Detect layer in yaml");
}

int Yolo10Impl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params  = this->named_parameters(true);
  auto buffers = this->named_buffers(true);
  int n = 0;
  for (const auto& e : entries) {
    if (auto* p = params.find(e.first)) {
      if (p->sizes() != e.second.sizes()) continue;
      torch::NoGradGuard ng;
      p->copy_(e.second.to(p->device(), p->dtype()));
      ++n;
    } else if (auto* b = buffers.find(e.first)) {
      if (b->sizes() != e.second.sizes()) continue;
      torch::NoGradGuard ng;
      b->copy_(e.second.to(b->device(), b->dtype()));
      ++n;
    }
  }
  return n;
}

}  // namespace yolocpp::models
