#include "yolocpp/models/yolo11_tasks.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

// ─── Internal: v11 backbone+neck (the same yaml subset as Yolo11Detect's,
// but stopping at the layer before Detect — i.e. layers 0..22). Shared by
// all v11 task wrappers below.

namespace {

struct LSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};

const std::vector<LSpec>& v11_yaml_for_tasks() {
  // Mirrors v11_yaml() in yolo11.cpp but excludes the final Detect layer.
  // Each task module appends its own head (Segment/Pose/OBB) at idx 23.
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",  {64,   3, 2}},                  // 0
      {{-1}, "Conv",  {128,  3, 2}},                  // 1
      {{-1}, "C3k2",  {256,  2, 0, 25}},              // 2
      {{-1}, "Conv",  {256,  3, 2}},                  // 3
      {{-1}, "C3k2",  {512,  2, 0, 25}},              // 4
      {{-1}, "Conv",  {512,  3, 2}},                  // 5
      {{-1}, "C3k2",  {512,  2, 1, 50}},              // 6
      {{-1}, "Conv",  {1024, 3, 2}},                  // 7
      {{-1}, "C3k2",  {1024, 2, 1, 50}},              // 8
      {{-1}, "SPPF",  {1024, 5}},                     // 9
      {{-1}, "C2PSA", {1024, 2}},                     // 10
      {{-1},     "Upsample", {2}},                    // 11
      {{-1, 6},  "Concat",   {1}},                    // 12
      {{-1},     "C3k2",     {512,  2, 0, 50}},       // 13
      {{-1},     "Upsample", {2}},                    // 14
      {{-1, 4},  "Concat",   {1}},                    // 15
      {{-1},     "C3k2",     {256,  2, 0, 50}},       // 16
      {{-1},     "Conv",     {256,  3, 2}},           // 17
      {{-1, 13}, "Concat",   {1}},                    // 18
      {{-1},     "C3k2",     {512,  2, 0, 50}},       // 19
      {{-1},     "Conv",     {512,  3, 2}},           // 20
      {{-1, 10}, "Concat",   {1}},                    // 21
      {{-1},     "C3k2",     {1024, 2, 1, 50}},       // 22
  };
  return y;
}

std::vector<int> build_v11_backbone_neck(torch::nn::ModuleList& model,
                                          Yolo11Scale scale,
                                          int img_in_ch = 3) {
  const auto& yaml = v11_yaml_for_tasks();
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
      int c_out = scale_channels_v11(s.a[0], scale);
      model->push_back(Conv(in_ch, c_out, s.a[1], s.a[2]));
      ch.push_back(c_out);
    } else if (s.kind == "C3k2") {
      int c_out = scale_channels_v11(s.a[0], scale);
      int n     = scale_depth_v11(s.a[1], scale);
      bool c3k  = (s.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;  // m/l/x force-upgrade
      double e  = (double)s.a[3] / 100.0;
      model->push_back(C3k2(in_ch, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s.kind == "SPPF") {
      int c_out = scale_channels_v11(s.a[0], scale);
      model->push_back(SPPF(in_ch, c_out, s.a[1]));
      ch.push_back(c_out);
    } else if (s.kind == "C2PSA") {
      int c_out = scale_channels_v11(s.a[0], scale);
      int n     = scale_depth_v11(s.a[1], scale);
      model->push_back(C2PSA(in_ch, c_out, n, /*e=*/0.5));
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

std::vector<torch::Tensor> forward_v11_backbone_neck(
    torch::nn::ModuleList& model, torch::Tensor x) {
  const auto& yaml = v11_yaml_for_tasks();
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
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
  return outs;
}

// Probe a 256×256 zero tensor through the backbone+neck and read off the
// strides at the three Detect inputs (layers 16, 19, 22).
std::vector<double> compute_v11_strides(torch::nn::ModuleList& model) {
  torch::NoGradGuard ng;
  auto x = torch::zeros({1, 3, 256, 256});
  auto outs = forward_v11_backbone_neck(model, x);
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
      ss << "v11 task load: shape mismatch for " << k << " ours="
         << dst.sizes() << " ckpt=" << t.sizes();
      throw std::runtime_error(ss.str());
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("v11 task load: copied 0 tensors");
  return copied;
}

// Remap state_dict keys: insert ".detect." after "model.23." for cv2/cv3/dfl
// (Upstream Segment/Pose/OBB inherit Detect, but our nesting wraps Detect
// inside the task head). cv4/proto stay where they are.
std::vector<std::pair<std::string, at::Tensor>>
remap_task_keys(const std::vector<std::pair<std::string, at::Tensor>>& entries,
                const std::string& head_idx /* "23" for v11 detect-style tasks */) {
  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(entries.size());
  std::string head = "model." + head_idx + ".";
  for (const auto& [k, t] : entries) {
    std::string nk = k;
    if (k.rfind(head, 0) == 0) {
      auto sub = k.substr(head.size());
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0 ||
          sub.rfind("dfl.", 0) == 0) {
        nk = head + "detect." + sub;
      }
    }
    out.emplace_back(std::move(nk), t);
  }
  return out;
}

// Yolo11-cls backbone (10 layers; same as yolo11-cls.yaml — no SPPF, C2PSA
// directly after the deepest C3k2).
struct LSpecCls { std::string kind; std::vector<int> a; };
const std::vector<LSpecCls>& v11_cls_yaml() {
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

// ─── Yolo11SegmentImpl ────────────────────────────────────────────────────
Yolo11SegmentImpl::Yolo11SegmentImpl(Yolo11Scale s, int nc_, int nm,
                                       int npr_unscaled)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v11_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  // Reuse the v8 task module-list, but cast scale into Yolo8Scale shape so
  // that scale_channels (used internally by SegmentImpl for `npr`) gives
  // the v11-correct value. We forge a Yolo8Scale with the same width.
  Yolo8Scale shim_scale{scale.depth_multiple, scale.width_multiple,
                        scale.max_channels};
  auto seg = Segment(nc, nm, npr_unscaled, det_ch, shim_scale, /*legacy=*/false);
  model->push_back(seg);
  stride = compute_v11_strides(model);
  seg->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo11SegmentImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v11_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* seg = model[23]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward(det_in);
}

int Yolo11SegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo11PoseImpl ───────────────────────────────────────────────────────
Yolo11PoseImpl::Yolo11PoseImpl(Yolo11Scale s, int nc_, int num_kpts_,
                                 int kpt_dim_)
    : scale(s), nc(nc_), num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v11_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  auto pose = Pose(nc, num_kpts, kpt_dim, det_ch, /*legacy=*/false);
  model->push_back(pose);
  stride = compute_v11_strides(model);
  pose->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo11PoseImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v11_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* p = model[23]->as<PoseImpl>();
  p->stride = stride;
  return p->forward(det_in);
}

int Yolo11PoseImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo11OBBImpl ────────────────────────────────────────────────────────
Yolo11OBBImpl::Yolo11OBBImpl(Yolo11Scale s, int nc_, int ne_)
    : scale(s), nc(nc_), ne(ne_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v11_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[16], ch[19], ch[22]};
  auto obb = OBB(nc, ne, det_ch, /*legacy=*/false);
  model->push_back(obb);
  stride = compute_v11_strides(model);
  obb->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo11OBBImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v11_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[16], outs[19], outs[22]};
  auto* o = model[23]->as<OBBImpl>();
  o->stride = stride;
  return o->forward(det_in);
}

int Yolo11OBBImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "23");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo11ClassifyImpl ───────────────────────────────────────────────────
Yolo11ClassifyImpl::Yolo11ClassifyImpl(Yolo11Scale s, int nc_)
    : scale(s), nc(nc_) {
  // Classify models use BN eps=1e-5 (not the 1e-3 detect override) — see
  // yolo8.cpp's BnEpsScope and the matching note in yolo8_classify.cpp.
  BnEpsScope eps_scope(1e-5);
  model = register_module("model", torch::nn::ModuleList());
  const auto& y = v11_cls_yaml();
  std::vector<int> ch;
  int c_in = 3;
  for (size_t i = 0; i < y.size(); ++i) {
    const auto& s2 = y[i];
    int prev = (i == 0) ? c_in : ch.back();
    if (s2.kind == "Conv") {
      int c_out = scale_channels_v11(s2.a[0], scale);
      model->push_back(Conv(prev, c_out, s2.a[1], s2.a[2]));
      ch.push_back(c_out);
    } else if (s2.kind == "C3k2") {
      int c_out = scale_channels_v11(s2.a[0], scale);
      int n     = scale_depth_v11(s2.a[1], scale);
      bool c3k  = (s2.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;  // m/l/x force-upgrade
      double e  = (double)s2.a[3] / 100.0;
      model->push_back(C3k2(prev, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s2.kind == "C2PSA") {
      int c_out = scale_channels_v11(s2.a[0], scale);
      int n     = scale_depth_v11(s2.a[1], scale);
      model->push_back(C2PSA(prev, c_out, n, /*e=*/0.5));
      ch.push_back(c_out);
    } else if (s2.kind == "Classify") {
      // Classify head reuses the v8 ClassifyImpl module (Conv c_in→1280 +
      // AdaptiveAvgPool1 + Linear 1280→nc).
      model->push_back(Classify(prev, nc, /*c_hidden=*/1280));
      ch.push_back(nc);
    }
  }
}

torch::Tensor Yolo11ClassifyImpl::forward(torch::Tensor x) {
  for (size_t i = 0; i < model->size(); ++i) {
    if (auto m = model[i]->as<ConvImpl>())          x = m->forward(x);
    else if (auto m = model[i]->as<C3k2Impl>())     x = m->forward(x);
    else if (auto m = model[i]->as<C2PSAImpl>())    x = m->forward(x);
    else if (auto m = model[i]->as<ClassifyImpl>()) x = m->forward(x);
  }
  return x;
}

int Yolo11ClassifyImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, entries);
}

}  // namespace yolocpp::models
