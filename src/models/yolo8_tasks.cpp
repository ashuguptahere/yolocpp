#include "yolocpp/models/yolo8_tasks.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

// ─── Proto ────────────────────────────────────────────────────────────────
ProtoImpl::ProtoImpl(int c1, int c_, int c2) {
  cv1      = register_module("cv1", Conv(c1, c_, 3));
  // Upstream: ConvTranspose2d(c_, c_, 2, 2, 0)
  upsample = register_module(
      "upsample",
      torch::nn::ConvTranspose2d(
          torch::nn::ConvTranspose2dOptions(c_, c_, 2).stride(2).padding(0)));
  cv2 = register_module("cv2", Conv(c_, c_, 3));
  cv3 = register_module("cv3", Conv(c_, c2, 1));
}

torch::Tensor ProtoImpl::forward(torch::Tensor x) {
  return cv3(cv2(upsample(cv1(x))));
}

// Helper: build a per-level cv4 ModuleList of length nl.
// Each entry: Sequential(Conv(c_in, c_inner, 3), Conv(c_inner, c_inner, 3),
//                        Conv2d(c_inner, c_out, 1))
static torch::nn::ModuleList build_cv4(const std::vector<int>& ch,
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

// ─── SegmentImpl ──────────────────────────────────────────────────────────
SegmentImpl::SegmentImpl(int nc_, int nm_, int npr_unscaled,
                         std::vector<int> ch_,
                         const Yolo8Scale& sc, bool legacy)
    : nm(nm_), npr(scale_channels(npr_unscaled, sc)),
      nl((int)ch_.size()), nc(nc_), reg_max(16), ch(std::move(ch_)) {
  // Inner Detect handles cv2/cv3/dfl. legacy=false picks up v11's
  // DWConv-Conv cv3 form.
  detect = register_module("detect", Detect(nc, ch, legacy));
  // cv4: mask-coefficient branches per level.
  int c4 = std::max(ch[0] / 4, nm);
  cv4    = register_module("cv4", build_cv4(ch, c4, nm));
  // Proto: takes the smallest-stride feature (P3) and produces prototypes.
  proto  = register_module("proto", Proto(ch[0], npr, nm));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SegmentImpl::forward(std::vector<torch::Tensor> x) {
  // Run Detect head — produces concat-form per-level features and decoded.
  detect->stride = stride;
  auto feats = detect->forward_features(x);
  auto decoded = detect->decode(feats);  // [N, 4 + nc, A]

  // Mask coefficients per level → flatten + concat → [N, nm, A]
  std::vector<torch::Tensor> mc;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);                     // [N, nm, h_i, w_i]
    auto N    = y.size(0);
    mc.push_back(y.reshape({N, nm, -1}));
  }
  auto coefs = torch::cat(mc, /*dim=*/2);

  // Prototypes from P3 (the highest-resolution detect input).
  auto protos = proto->forward(x[0]);                   // [N, nm, h_p, w_p]

  return {decoded, coefs, protos};
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
SegmentImpl::forward_train(std::vector<torch::Tensor> x) {
  // Same as forward() but returns the RAW per-level detect features (no
  // decode) so the detect loss can run on them (§5). Mask coefs + protos
  // are computed identically.
  detect->stride = stride;
  auto feats = detect->forward_features(x);   // per-level [N, no, h_i, w_i]

  std::vector<torch::Tensor> mc;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);                     // [N, nm, h_i, w_i]
    mc.push_back(y.reshape({y.size(0), nm, -1}));
  }
  auto coefs  = torch::cat(mc, /*dim=*/2);              // [N, nm, A]
  auto protos = proto->forward(x[0]);                   // [N, nm, h_p, w_p]
  return {feats, coefs, protos};
}

// ─── PoseImpl ─────────────────────────────────────────────────────────────
PoseImpl::PoseImpl(int nc_, int num_kpts_, int kpt_dim_,
                   std::vector<int> ch_, bool legacy)
    : nl((int)ch_.size()), nc(nc_), reg_max(16), ch(std::move(ch_)),
      num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  nk = num_kpts * kpt_dim;
  detect = register_module("detect", Detect(nc, ch, legacy));
  int c4 = std::max(ch[0] / 4, nk);
  cv4    = register_module("cv4", build_cv4(ch, c4, nk));
}

std::tuple<torch::Tensor, torch::Tensor>
PoseImpl::forward(std::vector<torch::Tensor> x) {
  detect->stride = stride;
  auto feats   = detect->forward_features(x);
  auto decoded = detect->decode(feats);  // [N, 4 + nc, A]

  // Keypoint regression per level → [N, nk, A]
  std::vector<torch::Tensor> ks;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);                  // [N, nk, h_i, w_i]
    auto N    = y.size(0);
    ks.push_back(y.reshape({N, nk, -1}));
  }
  auto raw = torch::cat(ks, 2);                      // [N, nk, A]

  // Decode: per-anchor keypoint = (raw_xy * 2 + (anchor_xy - 0.5)) * stride
  // confidence channel → sigmoid.
  auto N = raw.size(0);
  auto A = raw.size(2);
  auto kpts = raw.reshape({N, num_kpts, kpt_dim, A});

  // Build per-anchor stride and anchor xy in pixels (mirror the Detect path).
  auto opts = raw.options();
  std::vector<torch::Tensor> str_t;
  std::vector<torch::Tensor> anc;
  // Reconstruct from feat sizes: each cv4 level had h_i*w_i anchors.
  std::vector<int64_t> hwi;
  for (int i = 0; i < nl; ++i) {
    int h = (int)x[i].size(2), w = (int)x[i].size(3);
    hwi.push_back(h * w);
    // Anchor xy in PIXELS = (cell + 0.5) * stride. The decode below
    // (`xy*2*stride + (anchor_pix - 0.5*stride)`) needs pixels — building it
    // in feature units (no *stride) compressed every keypoint toward the
    // origin and disagreed with the (correct) ONNX emitter.
    auto sx = (torch::arange(w, opts) + 0.5) * stride[i];
    auto sy = (torch::arange(h, opts) + 0.5) * stride[i];
    auto gx = sx.reshape({1, w}).expand({h, w});
    auto gy = sy.reshape({h, 1}).expand({h, w});
    auto a  = torch::stack({gx, gy}, -1).reshape({h * w, 2});
    anc.push_back(a);
    str_t.push_back(torch::full({h * w}, stride[i], opts));
  }
  auto anc_full = torch::cat(anc, 0);                // [A, 2]
  auto str_full = torch::cat(str_t, 0);              // [A]

  // Apply upstream decode.
  //   In feature units: kpts_xy_feat = xy * 2 + (anchor_feat - 0.5)
  //                     kpts_xy_pix  = kpts_xy_feat * stride
  //   With anchor_feat = cell_idx + 0.5 (make_anchors offset),
  //                     kpts_xy_pix  = (xy * 2 + cell_idx) * stride
  //                                  = xy * 2 * stride + cell_idx * stride
  //                                  = xy * 2 * stride + (anchor_pix - 0.5*stride)
  // (Earlier code had `(xy*2 - 1)*stride + anchor_pix` which is off by
  //  −0.5*stride per element — produced 4–16 pixel keypoint offsets vs
  //  upstream depending on level. Caught by the ONNX-vs-Python parity
  //  comparator.)
  auto xy   = kpts.slice(2, 0, 2);                   // [N, K, 2, A]
  auto conf = kpts.slice(2, 2, 3);                   // [N, K, 1, A]
  auto anc_b = anc_full.transpose(0, 1).unsqueeze(0).unsqueeze(0);  // [1,1,2,A]
  auto str_b = str_full.unsqueeze(0).unsqueeze(0).unsqueeze(0);     // [1,1,1,A]
  auto xy_pix = xy * 2.0 * str_b + (anc_b - 0.5 * str_b);
  auto conf_s = conf.sigmoid();
  auto kpts_dec = torch::cat({xy_pix, conf_s}, /*dim=*/2);  // [N, K, 3, A]

  return {decoded, kpts_dec.reshape({N, nk, A})};
}

// ─── OBBImpl ──────────────────────────────────────────────────────────────
OBBImpl::OBBImpl(int nc_, int ne_, std::vector<int> ch_, bool legacy)
    : ne(ne_), nl((int)ch_.size()), nc(nc_), reg_max(16),
      ch(std::move(ch_)) {
  detect = register_module("detect", Detect(nc, ch, legacy));
  int c4 = std::max(ch[0] / 4, ne * 4);
  cv4    = register_module("cv4", build_cv4(ch, c4, ne));
}

std::tuple<torch::Tensor, torch::Tensor>
OBBImpl::forward(std::vector<torch::Tensor> x) {
  detect->stride = stride;
  auto feats = detect->forward_features(x);

  // ── Angle branch: cv4 + sigmoid + shift to [-π/4, 3π/4] ────────────────
  std::vector<torch::Tensor> as;
  for (int i = 0; i < nl; ++i) {
    auto* seq = cv4[i]->as<torch::nn::SequentialImpl>();
    auto y    = seq->forward(x[i]);
    auto N    = y.size(0);
    as.push_back(y.reshape({N, ne, -1}));
  }
  auto angle = torch::cat(as, /*dim=*/2);             // [N, ne=1, A]
  angle = (angle.sigmoid() - 0.25) * M_PI;            // [N, 1, A]

  // ── Rotated-box decode (upstream dist2rbox) ────────────────────────────
  // Per-anchor: lt, rb in feature units (DFL expectation), then
  //   xf = (r - l)/2,  yf = (b - t)/2
  //   cx_feat = xf*cos − yf*sin + anchor_x_feat
  //   cy_feat = xf*sin + yf*cos + anchor_y_feat
  //   w_feat  = l + r,   h_feat = t + b
  // Multiply by stride → pixels; convert to xyxy (the predictor reverses
  // back to the rotated rect via the angle).
  std::vector<torch::Tensor> flat_pred;
  std::vector<torch::Tensor> anchors_feat;            // (cell+0.5) per A
  std::vector<torch::Tensor> str_t;                   // [A]
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
  auto pred = torch::cat(flat_pred, /*dim=*/2);                 // [N, no, A]
  auto anc  = torch::cat(anchors_feat, /*dim=*/0);              // [A, 2]
  auto strd = torch::cat(str_t,         /*dim=*/0);             // [A]

  auto box = pred.slice(/*dim=*/1, 0,           4 * reg_max);
  auto cls = pred.slice(/*dim=*/1, 4 * reg_max, detect->no);
  auto dist = detect->dfl(box);                                  // [N, 4, A] (feature units)
  auto lt = dist.slice(/*dim=*/1, 0, 2);                         // [N, 2, A]
  auto rb = dist.slice(/*dim=*/1, 2, 4);

  auto xf = (rb.select(1, 0) - lt.select(1, 0)) * 0.5;           // [N, A]
  auto yf = (rb.select(1, 1) - lt.select(1, 1)) * 0.5;
  auto a_sq = angle.squeeze(1);                                  // [N, A]
  auto cos_a = a_sq.cos();
  auto sin_a = a_sq.sin();
  auto anc_x = anc.select(1, 0).unsqueeze(0);                    // [1, A]
  auto anc_y = anc.select(1, 1).unsqueeze(0);
  auto cx_feat = xf * cos_a - yf * sin_a + anc_x;                // [N, A]
  auto cy_feat = xf * sin_a + yf * cos_a + anc_y;
  auto w_feat  = lt.select(1, 0) + rb.select(1, 0);              // [N, A]
  auto h_feat  = lt.select(1, 1) + rb.select(1, 1);

  auto cx_pix = cx_feat * strd.unsqueeze(0);                     // [N, A]
  auto cy_pix = cy_feat * strd.unsqueeze(0);
  auto w_pix  = w_feat  * strd.unsqueeze(0);
  auto h_pix  = h_feat  * strd.unsqueeze(0);
  auto x1 = cx_pix - w_pix * 0.5;
  auto y1 = cy_pix - h_pix * 0.5;
  auto x2 = cx_pix + w_pix * 0.5;
  auto y2 = cy_pix + h_pix * 0.5;
  auto xyxy = torch::stack({x1, y1, x2, y2}, /*dim=*/1);         // [N, 4, A]

  cls = cls.sigmoid();
  auto decoded = torch::cat({xyxy, cls}, /*dim=*/1);             // [N, 4+nc, A]
  return {decoded, angle.squeeze(1)};
}

// ─── Shared backbone+neck builder for layers 0..21 ────────────────────────
//
// Identical to the v8 detect YAML connectivity.

namespace {
struct LSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};
const std::vector<LSpec>& v8_yaml_for_tasks() {
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",     {64,  3, 2}}, {{-1}, "Conv",     {128, 3, 2}},
      {{-1}, "C2f",      {128, 3, 1}}, {{-1}, "Conv",     {256, 3, 2}},
      {{-1}, "C2f",      {256, 6, 1}}, {{-1}, "Conv",     {512, 3, 2}},
      {{-1}, "C2f",      {512, 6, 1}}, {{-1}, "Conv",     {1024,3, 2}},
      {{-1}, "C2f",      {1024,3, 1}}, {{-1}, "SPPF",     {1024,5}},
      {{-1},     "Upsample", {2}},                {{-1, 6},  "Concat", {1}},
      {{-1},     "C2f",      {512, 3, 0}},        {{-1},     "Upsample", {2}},
      {{-1, 4},  "Concat",   {1}},                {{-1},     "C2f",      {256, 3, 0}},
      {{-1},     "Conv",     {256, 3, 2}},        {{-1, 12}, "Concat",   {1}},
      {{-1},     "C2f",      {512, 3, 0}},        {{-1},     "Conv",     {512, 3, 2}},
      {{-1, 9},  "Concat",   {1}},                {{-1},     "C2f",      {1024,3, 0}},
  };
  return y;
}

// Build layers 0..21 into an existing ModuleList. Returns per-layer output
// channels.
std::vector<int> build_backbone_neck(torch::nn::ModuleList& model,
                                     Yolo8Scale scale, int img_in_ch = 3) {
  const auto& yaml = v8_yaml_for_tasks();
  std::vector<int> ch;
  int c_in = img_in_ch;
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& spec = yaml[i];
    int in_ch = 0;
    if (spec.kind == "Concat") {
      for (int f : spec.from) {
        int idx = (f == -1) ? (int)i - 1 : f;
        in_ch += ch[idx];
      }
    } else {
      int f = spec.from[0];
      int idx = (f == -1) ? (int)i - 1 : f;
      in_ch = (idx == -1) ? c_in : ch[idx];
    }
    if (spec.kind == "Conv") {
      int c_out = scale_channels(spec.a[0], scale);
      model->push_back(Conv(in_ch, c_out, spec.a[1], spec.a[2]));
      ch.push_back(c_out);
    } else if (spec.kind == "C2f") {
      int c_out = scale_channels(spec.a[0], scale);
      int n     = scale_depth(spec.a[1], scale);
      bool sc   = (spec.a[2] != 0);
      model->push_back(C2f(in_ch, c_out, n, sc));
      ch.push_back(c_out);
    } else if (spec.kind == "SPPF") {
      int c_out = scale_channels(spec.a[0], scale);
      model->push_back(SPPF(in_ch, c_out, spec.a[1]));
      ch.push_back(c_out);
    } else if (spec.kind == "Upsample") {
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{(double)spec.a[0],
                                                (double)spec.a[0]})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (spec.kind == "Concat") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    }
  }
  return ch;
}

// Forward layers 0..21 of the model, returning the per-layer outputs vector
// (size 22). Caller uses outs[15], outs[18], outs[21] as detect-head inputs.
std::vector<torch::Tensor> forward_backbone_neck(
    torch::nn::ModuleList& model, torch::Tensor x) {
  const auto& yaml = v8_yaml_for_tasks();
  std::vector<torch::Tensor> outs(yaml.size());
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& spec = yaml[i];
    torch::Tensor in;
    if (spec.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : spec.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
      in = torch::cat(parts, 1);
    } else {
      int f = spec.from[0];
      in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
    }
    if (spec.kind == "Conv")          outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (spec.kind == "C2f")      outs[i] = model[i]->as<C2fImpl>()->forward(in);
    else if (spec.kind == "SPPF")     outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (spec.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (spec.kind == "Concat")   outs[i] = in;
  }
  return outs;
}

// Compute strides for the three detect inputs (15, 18, 21) by probing.
std::vector<double> compute_strides(torch::nn::ModuleList& model) {
  torch::NoGradGuard ng;
  auto x   = torch::zeros({1, 3, 256, 256});
  auto outs = forward_backbone_neck(model, x);
  std::vector<double> str = {
      256.0 / outs[15].size(2),
      256.0 / outs[18].size(2),
      256.0 / outs[21].size(2),
  };
  return str;
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
  int copied = 0, skipped_shape = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    auto& dst = it->second;
    if (dst.sizes() != t.sizes()) {
      ++skipped_shape;
      continue;
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("load: copied 0 tensors");
  if (skipped_shape > 0)
    std::cerr << "[v8-task load] skipped " << skipped_shape
              << " tensors with shape mismatch (head re-purposed for custom nc)\n";
  return copied;
}

}  // anonymous namespace

// ─── Yolo8SegmentImpl ────────────────────────────────────────────────────
Yolo8SegmentImpl::Yolo8SegmentImpl(Yolo8Scale s, int nc_, int nm,
                                      int npr_unscaled)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[15], ch[18], ch[21]};
  // Note: `scale` arg shadows member; pass `s`.
  auto seg = Segment(nc, nm, npr_unscaled, det_ch, s);
  model->push_back(seg);
  stride = compute_strides(model);
  seg->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo8SegmentImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[15], outs[18], outs[21]};
  auto* seg = model[22]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward(det_in);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
Yolo8SegmentImpl::forward_train_seg(torch::Tensor x) {
  auto outs = forward_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[15], outs[18], outs[21]};
  auto* seg = model[22]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward_train(det_in);
}

int Yolo8SegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  // Upstream state_dict has model.22.cv2/cv3/dfl/cv4/proto. Our nesting
  // is model.22.detect.cv2 etc. — we need to remap the prefix.
  std::vector<std::pair<std::string, at::Tensor>> remapped;
  remapped.reserve(entries.size());
  for (const auto& [k, t] : entries) {
    std::string nk = k;
    static const std::string head = "model.22.";
    if (k.rfind(head, 0) == 0) {
      auto sub = k.substr(head.size());
      // detect-owned keys: cv2, cv3, dfl
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0 ||
          sub.rfind("dfl.", 0) == 0) {
        nk = head + "detect." + sub;
      }
      // cv4 and proto stay as-is (they live directly on Segment).
    }
    remapped.emplace_back(std::move(nk), t);
  }
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo8PoseImpl ───────────────────────────────────────────────────────
Yolo8PoseImpl::Yolo8PoseImpl(Yolo8Scale s, int nc_, int num_kpts_, int kpt_dim_)
    : scale(s), nc(nc_), num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[15], ch[18], ch[21]};
  auto pose = Pose(nc, num_kpts, kpt_dim, det_ch);
  model->push_back(pose);
  stride = compute_strides(model);
  pose->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo8PoseImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[15], outs[18], outs[21]};
  auto* p = model[22]->as<PoseImpl>();
  p->stride = stride;
  return p->forward(det_in);
}

int Yolo8PoseImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  std::vector<std::pair<std::string, at::Tensor>> remapped;
  remapped.reserve(entries.size());
  for (const auto& [k, t] : entries) {
    std::string nk = k;
    static const std::string head = "model.22.";
    if (k.rfind(head, 0) == 0) {
      auto sub = k.substr(head.size());
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0 ||
          sub.rfind("dfl.", 0) == 0) {
        nk = head + "detect." + sub;
      }
    }
    remapped.emplace_back(std::move(nk), t);
  }
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo8OBBImpl ────────────────────────────────────────────────────────
Yolo8OBBImpl::Yolo8OBBImpl(Yolo8Scale s, int nc_, int ne_)
    : scale(s), nc(nc_), ne(ne_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[15], ch[18], ch[21]};
  auto obb = OBB(nc, ne, det_ch);
  model->push_back(obb);
  stride = compute_strides(model);
  obb->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo8OBBImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[15], outs[18], outs[21]};
  auto* o = model[22]->as<OBBImpl>();
  o->stride = stride;
  return o->forward(det_in);
}

int Yolo8OBBImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  std::vector<std::pair<std::string, at::Tensor>> remapped;
  remapped.reserve(entries.size());
  for (const auto& [k, t] : entries) {
    std::string nk = k;
    static const std::string head = "model.22.";
    if (k.rfind(head, 0) == 0) {
      auto sub = k.substr(head.size());
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0 ||
          sub.rfind("dfl.", 0) == 0) {
        nk = head + "detect." + sub;
      }
    }
    remapped.emplace_back(std::move(nk), t);
  }
  return load_state_dict_generic(*this, remapped);
}

}  // namespace yolocpp::models
