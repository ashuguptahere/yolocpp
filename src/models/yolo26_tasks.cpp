#include "yolocpp/models/yolo26_tasks.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

namespace F = torch::nn::functional;

// ─── Helper: build per-level cv4 (Conv→Conv→Conv2d) ───────────────────────

static torch::nn::ModuleList build_cv4_v26(const std::vector<int>& ch,
                                            int c_inner, int c_out) {
  auto ml = torch::nn::ModuleList();
  for (size_t i = 0; i < ch.size(); ++i) {
    auto seq = torch::nn::Sequential();
    seq->push_back(Conv(ch[i],   c_inner, 3));
    seq->push_back(Conv(c_inner, c_inner, 3));
    seq->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c_inner, c_out, 1)));
    ml->push_back(seq);
  }
  return ml;
}

// ─── Segment26Impl ────────────────────────────────────────────────────────

Segment26Impl::Segment26Impl(int nc_, int nm_, int npr_unscaled,
                              std::vector<int> ch_,
                              const Yolo26Scale& sc)
    : nm(nm_), npr(scale_channels_v26(npr_unscaled, sc)),
      nl((int)ch_.size()), nc(nc_), ch(std::move(ch_)) {
  detect = register_module("detect", Detect26(nc, ch));
  int c4 = std::max(ch[0] / 4, nm);
  cv4    = register_module("cv4", build_cv4_v26(ch, c4, nm));
  proto  = register_module("proto", Proto(ch[0], npr, nm));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Segment26Impl::forward(std::vector<torch::Tensor> x) {
  detect->stride = stride;
  auto feats   = detect->forward_features(x);
  auto decoded = detect->decode(feats);     // [N, 4 + nc, A]

  std::vector<torch::Tensor> mc;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);
    auto N    = y.size(0);
    mc.push_back(y.reshape({N, nm, -1}));
  }
  auto coefs  = torch::cat(mc, /*dim=*/2);
  auto protos = proto->forward(x[0]);

  return {decoded, coefs, protos};
}

// ─── Pose26Impl ───────────────────────────────────────────────────────────

Pose26Impl::Pose26Impl(int nc_, int num_kpts_, int kpt_dim_,
                       std::vector<int> ch_, int nk_sigma_)
    : nk_sigma(nk_sigma_), nl((int)ch_.size()), nc(nc_), ch(std::move(ch_)),
      num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  nk = num_kpts * kpt_dim;
  detect = register_module("detect", Detect26(nc, ch));
  int total = nk + nk_sigma;                  // v26 cv4 emits both kpts + σ
  int c4 = std::max(ch[0] / 4, total);
  cv4    = register_module("cv4", build_cv4_v26(ch, c4, total));
}

std::tuple<torch::Tensor, torch::Tensor>
Pose26Impl::forward(std::vector<torch::Tensor> x) {
  detect->stride = stride;
  auto feats   = detect->forward_features(x);
  auto decoded = detect->decode(feats);

  int total = nk + nk_sigma;
  std::vector<torch::Tensor> ks;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);
    auto N    = y.size(0);
    ks.push_back(y.reshape({N, total, -1}));
  }
  auto full = torch::cat(ks, 2);                     // [N, nk+nk_sigma, A]
  auto raw  = full.slice(/*dim=*/1, 0, nk);          // [N, nk, A] — drop σ for inference

  auto N = raw.size(0);
  auto A = raw.size(2);
  auto kpts = raw.reshape({N, num_kpts, kpt_dim, A});

  auto opts = raw.options();
  std::vector<torch::Tensor> str_t;
  std::vector<torch::Tensor> anc;
  for (int i = 0; i < nl; ++i) {
    int h = (int)x[i].size(2), w = (int)x[i].size(3);
    auto sx = (torch::arange(w, opts) + 0.5);
    auto sy = (torch::arange(h, opts) + 0.5);
    auto gx = sx.reshape({1, w}).expand({h, w});
    auto gy = sy.reshape({h, 1}).expand({h, w});
    auto a  = torch::stack({gx, gy}, -1).reshape({h * w, 2});
    anc.push_back(a);
    str_t.push_back(torch::full({h * w}, stride[i], opts));
  }
  auto anc_full = torch::cat(anc, 0);
  auto str_full = torch::cat(str_t, 0);

  auto xy   = kpts.slice(2, 0, 2);
  auto conf = kpts.slice(2, 2, 3);
  auto anc_b = anc_full.transpose(0, 1).unsqueeze(0).unsqueeze(0);
  auto str_b = str_full.unsqueeze(0).unsqueeze(0).unsqueeze(0);
  // Upstream formula: kpts_pix = (xy * 2 + cell_idx) * stride
  //                              = xy*2*stride + (anchor_pix - 0.5*stride)
  // (See the matching note in yolo8_tasks.cpp for the bug history.)
  auto xy_pix = xy * 2.0 * str_b + (anc_b - 0.5 * str_b);
  auto conf_s = conf.sigmoid();
  auto kpts_dec = torch::cat({xy_pix, conf_s}, /*dim=*/2);

  return {decoded, kpts_dec.reshape({N, nk, A})};
}

// ─── OBB26Impl ────────────────────────────────────────────────────────────

OBB26Impl::OBB26Impl(int nc_, int ne_, std::vector<int> ch_)
    : ne(ne_), nl((int)ch_.size()), nc(nc_), ch(std::move(ch_)) {
  detect = register_module("detect", Detect26(nc, ch));
  int c4 = std::max(ch[0] / 4, ne * 4);
  cv4    = register_module("cv4", build_cv4_v26(ch, c4, ne));
}

std::tuple<torch::Tensor, torch::Tensor>
OBB26Impl::forward(std::vector<torch::Tensor> x) {
  detect->stride = stride;
  auto feats = detect->forward_features(x);

  // ── Angle branch ───────────────────────────────────────────────────────
  std::vector<torch::Tensor> as;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);
    auto N    = y.size(0);
    as.push_back(y.reshape({N, ne, -1}));
  }
  auto angle = torch::cat(as, /*dim=*/2);                    // [N, 1, A]
  angle = (angle.sigmoid() - 0.25) * M_PI;

  // ── Rotated decode (DFL-free; matches the upstream dist2rbox) ─────────
  std::vector<torch::Tensor> flat_pred;
  std::vector<torch::Tensor> anchors_feat, str_t;
  for (int i = 0; i < nl; ++i) {
    auto t = feats[i];
    auto N_ = t.size(0);
    auto h  = t.size(2);
    auto w  = t.size(3);
    flat_pred.push_back(t.view({N_, detect->no, h * w}));
    auto opts = torch::TensorOptions().device(t.device()).dtype(t.dtype());
    auto sy = torch::arange(h, opts) + 0.5;
    auto sx = torch::arange(w, opts) + 0.5;
    auto gy = sy.view({h, 1}).expand({h, w});
    auto gx = sx.view({1, w}).expand({h, w});
    anchors_feat.push_back(torch::stack({gx, gy}, -1).view({h * w, 2}));
    str_t.push_back(torch::full({h * w}, stride[i], opts));
  }
  auto pred = torch::cat(flat_pred, /*dim=*/2);              // [N, 4+nc, A]
  auto anc  = torch::cat(anchors_feat, /*dim=*/0);
  auto strd = torch::cat(str_t,         /*dim=*/0);

  auto box_raw = pred.slice(/*dim=*/1, 0, 4);                // [N, 4, A]
  auto cls     = pred.slice(/*dim=*/1, 4, detect->no);
  auto box_pos = torch::nn::functional::softplus(box_raw);   // distances ≥ 0
  auto lt = box_pos.slice(/*dim=*/1, 0, 2);
  auto rb = box_pos.slice(/*dim=*/1, 2, 4);

  auto xf = (rb.select(1, 0) - lt.select(1, 0)) * 0.5;
  auto yf = (rb.select(1, 1) - lt.select(1, 1)) * 0.5;
  auto a_sq = angle.squeeze(1);
  auto cos_a = a_sq.cos();
  auto sin_a = a_sq.sin();
  auto anc_x = anc.select(1, 0).unsqueeze(0);
  auto anc_y = anc.select(1, 1).unsqueeze(0);
  auto cx_feat = xf * cos_a - yf * sin_a + anc_x;
  auto cy_feat = xf * sin_a + yf * cos_a + anc_y;
  auto w_feat  = lt.select(1, 0) + rb.select(1, 0);
  auto h_feat  = lt.select(1, 1) + rb.select(1, 1);
  auto cx_pix = cx_feat * strd.unsqueeze(0);
  auto cy_pix = cy_feat * strd.unsqueeze(0);
  auto w_pix  = w_feat  * strd.unsqueeze(0);
  auto h_pix  = h_feat  * strd.unsqueeze(0);
  auto x1 = cx_pix - w_pix * 0.5;
  auto y1 = cy_pix - h_pix * 0.5;
  auto x2 = cx_pix + w_pix * 0.5;
  auto y2 = cy_pix + h_pix * 0.5;
  auto xyxy = torch::stack({x1, y1, x2, y2}, /*dim=*/1);
  cls = cls.sigmoid();
  auto decoded = torch::cat({xyxy, cls}, /*dim=*/1);
  return {decoded, angle.squeeze(1)};
}

// ─── Internal: shared v11 backbone+neck (re-used as v26's) ────────────────

namespace {

struct LSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};

const std::vector<LSpec>& v26_yaml_for_tasks() {
  // Mirrors v26_yaml() in yolo26.cpp but excludes the final Detect26 layer.
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",   {64,   3, 2}},
      {{-1}, "Conv",   {128,  3, 2}},
      {{-1}, "C3k2",   {256,  2, 0, 25}},
      {{-1}, "Conv",   {256,  3, 2}},
      {{-1}, "C3k2",   {512,  2, 0, 25}},
      {{-1}, "Conv",   {512,  3, 2}},
      {{-1}, "C3k2",   {512,  2, 1, 50}},
      {{-1}, "Conv",   {1024, 3, 2}},
      {{-1}, "C3k2",   {1024, 2, 1, 50}},
      {{-1}, "SPPF",   {1024, 5}},
      {{-1}, "C2PSA",  {1024, 2}},
      {{-1},     "Upsample", {2}},
      {{-1, 6},  "Concat",   {1}},
      {{-1},     "C3k2",     {512,  2, 1, 50}},   // c3k=True
      {{-1},     "Upsample", {2}},
      {{-1, 4},  "Concat",   {1}},
      {{-1},     "C3k2",     {256,  2, 1, 50}},   // c3k=True
      {{-1},     "Conv",     {256,  3, 2}},
      {{-1, 13}, "Concat",   {1}},
      {{-1},     "C3k2",     {512,  2, 1, 50}},   // c3k=True
      {{-1},     "Conv",     {512,  3, 2}},
      {{-1, 10}, "Concat",   {1}},
      {{-1},     "C2PSAf",   {1024, 1, 50}},      // new module, n=1
  };
  return y;
}

std::vector<int> build_v26_backbone_neck(torch::nn::ModuleList& model,
                                          Yolo26Scale scale,
                                          int img_in_ch = 3) {
  const auto& yaml = v26_yaml_for_tasks();
  std::vector<int> ch;
  int c_in = img_in_ch;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    int in_ch = 0;
    if (s.kind == "Concat") {
      for (int f : s.from) {
        int idx = (f == -1) ? (int)i - 1 : f;
        in_ch += ch[idx];
      }
    } else {
      int f = s.from[0];
      int idx = (f == -1) ? (int)i - 1 : f;
      in_ch = (idx == -1) ? c_in : ch[idx];
    }
    if (s.kind == "Conv") {
      int c_out = scale_channels_v26(s.a[0], scale);
      model->push_back(Conv(in_ch, c_out, s.a[1], s.a[2]));
      ch.push_back(c_out);
    } else if (s.kind == "C3k2") {
      int c_out = scale_channels_v26(s.a[0], scale);
      int n     = scale_depth_v26(s.a[1], scale);
      bool c3k  = (s.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;
      double e  = (double)s.a[3] / 100.0;
      model->push_back(C3k2(in_ch, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s.kind == "SPPF") {
      int c_out = scale_channels_v26(s.a[0], scale);
      // v26 SPPF: cv1 Identity-act + residual shortcut (see yolo26.cpp).
      model->push_back(SPPF(in_ch, c_out, s.a[1],
                            /*cv1_act=*/false, /*shortcut=*/true));
      ch.push_back(c_out);
    } else if (s.kind == "C2PSA") {
      int c_out = scale_channels_v26(s.a[0], scale);
      int n     = scale_depth_v26(s.a[1], scale);
      model->push_back(C2PSA(in_ch, c_out, n, /*e=*/0.5));
      ch.push_back(c_out);
    } else if (s.kind == "C2PSAf") {
      int c_out = scale_channels_v26(s.a[0], scale);
      int n     = scale_depth_v26(s.a[1], scale);
      double e  = (double)s.a[2] / 100.0;
      model->push_back(C2PSAf(in_ch, c_out, n, e));
      ch.push_back(c_out);
    } else if (s.kind == "Upsample") {
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{(double)s.a[0],
                                                (double)s.a[0]})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (s.kind == "Concat") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    }
  }
  return ch;
}

std::vector<torch::Tensor> forward_v26_backbone_neck(
    torch::nn::ModuleList& model, torch::Tensor x) {
  const auto& yaml = v26_yaml_for_tasks();
  std::vector<torch::Tensor> outs(yaml.size());
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
      in = torch::cat(parts, /*dim=*/1);
    } else {
      int f = s.from[0];
      in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
    }
    if      (s.kind == "Conv")     outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "C3k2")     outs[i] = model[i]->as<C3k2Impl>()->forward(in);
    else if (s.kind == "SPPF")     outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (s.kind == "C2PSA")    outs[i] = model[i]->as<C2PSAImpl>()->forward(in);
    else if (s.kind == "C2PSAf")   outs[i] = model[i]->as<C2PSAfImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
  return outs;
}

std::vector<double> compute_v26_strides(torch::nn::ModuleList& model) {
  torch::NoGradGuard ng;
  auto x = torch::zeros({1, 3, 256, 256});
  auto outs = forward_v26_backbone_neck(model, x);
  return {
      256.0 / (double)outs[16].size(2),
      256.0 / (double)outs[19].size(2),
      256.0 / (double)outs[22].size(2),
  };
}

template <typename M>
int load_state_dict_generic(M& self,
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params = self.named_parameters();
  auto buffs  = self.named_buffers();
  std::unordered_map<std::string, at::Tensor> ours;
  for (auto& kv : params) ours.emplace(kv.key(), kv.value());
  for (auto& kv : buffs)  ours.emplace(kv.key(), kv.value());
  torch::NoGradGuard ng;
  int copied = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    auto& dst = it->second;
    if (dst.sizes() != t.sizes()) {
      std::ostringstream ss;
      ss << "v26 task load: shape mismatch for " << k << " ours="
         << dst.sizes() << " ckpt=" << t.sizes();
      throw std::runtime_error(ss.str());
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("v26 task load: copied 0 tensors");
  return copied;
}

// Remap state_dict keys: insert ".detect." after "model.23." for cv2/cv3
// (the v8/v11 convention; v26 has no dfl key). cv4/proto stay where they are.
std::vector<std::pair<std::string, at::Tensor>>
remap_task_keys(const std::vector<std::pair<std::string, at::Tensor>>& entries,
                const std::string& head_idx) {
  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(entries.size());
  std::string head = "model." + head_idx + ".";
  for (const auto& [k, t] : entries) {
    std::string nk = k;
    if (k.rfind(head, 0) == 0) {
      auto sub = k.substr(head.size());
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0) {
        nk = head + "detect." + sub;
      }
    }
    out.emplace_back(std::move(nk), t);
  }
  return out;
}

// v26-cls backbone (10 layers + Classify head — same as v11-cls).
struct LSpecCls { std::string kind; std::vector<int> a; };
const std::vector<LSpecCls>& v26_cls_yaml() {
  static const std::vector<LSpecCls> y = {
      {"Conv",  {64,   3, 2}},
      {"Conv",  {128,  3, 2}},
      {"C3k2",  {256,  2, 0, 25}},
      {"Conv",  {256,  3, 2}},
      {"C3k2",  {512,  2, 0, 25}},
      {"Conv",  {512,  3, 2}},
      {"C3k2",  {512,  2, 1, 50}},
      {"Conv",  {1024, 3, 2}},
      {"C3k2",  {1024, 2, 1, 50}},
      {"C2PSA", {1024, 2}},
      {"Classify", {}},
  };
  return y;
}

}  // anonymous namespace

// ─── Yolo26SegmentImpl ────────────────────────────────────────────────────
Yolo26SegmentImpl::Yolo26SegmentImpl(Yolo26Scale s, int nc_, int nm,
                                       int npr_unscaled)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v26_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  auto seg = Segment26(nc, nm, npr_unscaled, det_ch, scale);
  model->push_back(seg);
  stride = compute_v26_strides(model);
  seg->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo26SegmentImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v26_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* seg = model[23]->as<Segment26Impl>();
  seg->stride = stride;
  return seg->forward(det_in);
}

int Yolo26SegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo26PoseImpl ───────────────────────────────────────────────────────
Yolo26PoseImpl::Yolo26PoseImpl(Yolo26Scale s, int nc_, int num_kpts_,
                                 int kpt_dim_)
    : scale(s), nc(nc_), num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v26_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  auto pose = Pose26(nc, num_kpts, kpt_dim, det_ch);
  model->push_back(pose);
  stride = compute_v26_strides(model);
  pose->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo26PoseImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v26_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* p = model[23]->as<Pose26Impl>();
  p->stride = stride;
  return p->forward(det_in);
}

int Yolo26PoseImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo26OBBImpl ────────────────────────────────────────────────────────
Yolo26OBBImpl::Yolo26OBBImpl(Yolo26Scale s, int nc_, int ne_)
    : scale(s), nc(nc_), ne(ne_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v26_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  auto obb = OBB26(nc, ne, det_ch);
  model->push_back(obb);
  stride = compute_v26_strides(model);
  obb->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo26OBBImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v26_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* o = model[23]->as<OBB26Impl>();
  o->stride = stride;
  return o->forward(det_in);
}

int Yolo26OBBImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo26ClassifyImpl ───────────────────────────────────────────────────
Yolo26ClassifyImpl::Yolo26ClassifyImpl(Yolo26Scale s, int nc_)
    : scale(s), nc(nc_) {
  // Classify models use BN eps=1e-5 (not the 1e-3 detect override) — see
  // yolo8.cpp's BnEpsScope and the matching note in yolo8_classify.cpp.
  BnEpsScope eps_scope(1e-5);
  model = register_module("model", torch::nn::ModuleList());
  const auto& y = v26_cls_yaml();
  std::vector<int> ch;
  int c_in = 3;
  for (size_t i = 0; i < y.size(); ++i) {
    const auto& s2 = y[i];
    int prev = (i == 0) ? c_in : ch.back();
    if (s2.kind == "Conv") {
      int c_out = scale_channels_v26(s2.a[0], scale);
      model->push_back(Conv(prev, c_out, s2.a[1], s2.a[2]));
      ch.push_back(c_out);
    } else if (s2.kind == "C3k2") {
      int c_out = scale_channels_v26(s2.a[0], scale);
      int n     = scale_depth_v26(s2.a[1], scale);
      bool c3k  = (s2.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;
      double e  = (double)s2.a[3] / 100.0;
      model->push_back(C3k2(prev, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s2.kind == "C2PSA") {
      int c_out = scale_channels_v26(s2.a[0], scale);
      int n     = scale_depth_v26(s2.a[1], scale);
      model->push_back(C2PSA(prev, c_out, n, /*e=*/0.5));
      ch.push_back(c_out);
    } else if (s2.kind == "Classify") {
      model->push_back(Classify(prev, nc, /*c_hidden=*/1280));
      ch.push_back(nc);
    }
  }
}

torch::Tensor Yolo26ClassifyImpl::forward(torch::Tensor x) {
  for (size_t i = 0; i < model->size(); ++i) {
    if      (auto m = model[i]->as<ConvImpl>())     x = m->forward(x);
    else if (auto m = model[i]->as<C3k2Impl>())     x = m->forward(x);
    else if (auto m = model[i]->as<C2PSAImpl>())    x = m->forward(x);
    else if (auto m = model[i]->as<ClassifyImpl>()) x = m->forward(x);
  }
  return x;
}

int Yolo26ClassifyImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, entries);
}

}  // namespace yolocpp::models
