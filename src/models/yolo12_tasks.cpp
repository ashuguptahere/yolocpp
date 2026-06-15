#include "yolocpp/models/yolo12_tasks.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

namespace {

struct LSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};

// Mirrors v12_yaml() in yolo12.cpp but stops one layer before Detect/
// Segment/Pose/OBB. The task wrapper appends its own head module at
// the final index.
const std::vector<LSpec>& v12_yaml_for_tasks() {
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",   {64,   3, 2}},
      {{-1}, "Conv",   {128,  3, 2}},
      {{-1}, "C3k2",   {256,  2, 0, 25}},
      {{-1}, "Conv",   {256,  3, 2}},
      {{-1}, "C3k2",   {512,  2, 0, 25}},
      {{-1}, "Conv",   {512,  3, 2}},
      {{-1}, "A2C2f",  {512,  4, 1, 4, 0, 200}},
      {{-1}, "Conv",   {1024, 3, 2}},
      {{-1}, "A2C2f",  {1024, 4, 1, 1, 0, 200}},
      {{-1},     "Upsample", {2}},
      {{-1, 6},  "Concat",   {1}},
      {{-1},     "A2C2f",    {512, 2, 0, -1, 0, 200}},
      {{-1},     "Upsample", {2}},
      {{-1, 4},  "Concat",   {1}},
      {{-1},     "A2C2f",    {256, 2, 0, -1, 0, 200}},
      {{-1},     "Conv",     {256,  3, 2}},
      {{-1, 11}, "Concat",   {1}},
      {{-1},     "A2C2f",    {512, 2, 0, -1, 0, 200}},
      {{-1},     "Conv",     {512,  3, 2}},
      {{-1, 8},  "Concat",   {1}},
      {{-1},     "C3k2",     {1024, 2, 1, 50}},
  };
  return y;
}

std::vector<int> build_v12_backbone_neck(torch::nn::ModuleList& model,
                                          Yolo12Scale scale) {
  const auto& yaml = v12_yaml_for_tasks();
  std::vector<int> ch;
  int c_in = 3;
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
      int c_out = scale_channels_v12(s.a[0], scale);
      model->push_back(Conv(in_ch, c_out, s.a[1], s.a[2]));
      ch.push_back(c_out);
    } else if (s.kind == "C3k2") {
      int c_out = scale_channels_v12(s.a[0], scale);
      int n     = scale_depth_v12(s.a[1], scale);
      bool c3k  = (s.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;
      double e  = (double)s.a[3] / 100.0;
      model->push_back(C3k2(in_ch, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s.kind == "A2C2f") {
      int c_out  = scale_channels_v12(s.a[0], scale);
      int n      = scale_depth_v12(s.a[1], scale);
      bool a2    = (s.a[2] != 0);
      int area   = s.a[3];
      bool resid = (s.a[4] != 0);
      double mlp = (double)s.a[5] / 100.0;
      if (scale.depth_multiple >= 1.0 && scale.width_multiple >= 1.0) {
        resid = true; mlp = 1.2;
      }
      model->push_back(A2C2f(in_ch, c_out, n, a2, area, resid, mlp, /*e=*/0.5));
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

std::vector<torch::Tensor> forward_v12_backbone_neck(
    torch::nn::ModuleList& model, torch::Tensor x) {
  const auto& yaml = v12_yaml_for_tasks();
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
    else if (s.kind == "A2C2f")    outs[i] = model[i]->as<A2C2fImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
  return outs;
}

std::vector<double> compute_v12_strides(torch::nn::ModuleList& model) {
  torch::NoGradGuard ng;
  auto x = torch::zeros({1, 3, 256, 256});
  auto outs = forward_v12_backbone_neck(model, x);
  return {
      256.0 / (double)outs[14].size(2),
      256.0 / (double)outs[17].size(2),
      256.0 / (double)outs[20].size(2),
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
  if (copied == 0) throw std::runtime_error("v12 task load: copied 0 tensors");
  if (skipped_shape > 0)
    std::cerr << "[v12-task load] skipped " << skipped_shape
              << " tensors with shape mismatch (head re-purposed for custom nc)\n";
  return copied;
}

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
      if (sub.rfind("cv2.", 0) == 0 || sub.rfind("cv3.", 0) == 0 ||
          sub.rfind("dfl.", 0) == 0) {
        nk = head + "detect." + sub;
      }
    }
    out.emplace_back(std::move(nk), t);
  }
  return out;
}

// v12-cls walker (mirrors v11-cls; ends in Classify). Layers 2,4 use
// halved C3k2 c_out to match the upstream yolo12-cls.yaml convention.
struct LSpecCls { std::string kind; std::vector<int> a; };
const std::vector<LSpecCls>& v12_cls_yaml() {
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
      {"Classify", {}},
  };
  return y;
}

}  // anonymous namespace

// ─── Yolo12SegmentImpl ────────────────────────────────────────────────────
Yolo12SegmentImpl::Yolo12SegmentImpl(Yolo12Scale s, int nc_, int nm,
                                       int npr_unscaled)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v12_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[14], ch[17], ch[20]};
  // Forge a Yolo8Scale shim with the v12 width — Segment uses it for its
  // npr scaling.
  Yolo8Scale shim_scale{scale.depth_multiple, scale.width_multiple,
                         scale.max_channels};
  auto seg = Segment(nc, nm, npr_unscaled, det_ch, shim_scale, /*legacy=*/false);
  model->push_back(seg);
  stride = compute_v12_strides(model);
  seg->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo12SegmentImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v12_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[14], outs[17], outs[20]};
  auto* seg = model[21]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward(det_in);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
Yolo12SegmentImpl::forward_train_seg(torch::Tensor x) {
  auto outs = forward_v12_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[14], outs[17], outs[20]};
  auto* seg = model[21]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward_train(det_in);
}

int Yolo12SegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "21");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo12PoseImpl ───────────────────────────────────────────────────────
Yolo12PoseImpl::Yolo12PoseImpl(Yolo12Scale s, int nc_, int num_kpts_,
                                 int kpt_dim_)
    : scale(s), nc(nc_), num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v12_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[14], ch[17], ch[20]};
  auto pose = Pose(nc, num_kpts, kpt_dim, det_ch, /*legacy=*/false);
  model->push_back(pose);
  stride = compute_v12_strides(model);
  pose->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo12PoseImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v12_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[14], outs[17], outs[20]};
  auto* p = model[21]->as<PoseImpl>();
  p->stride = stride;
  return p->forward(det_in);
}

int Yolo12PoseImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "21");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo12OBBImpl ────────────────────────────────────────────────────────
Yolo12OBBImpl::Yolo12OBBImpl(Yolo12Scale s, int nc_, int ne_)
    : scale(s), nc(nc_), ne(ne_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v12_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[14], ch[17], ch[20]};
  auto obb = OBB(nc, ne, det_ch, /*legacy=*/false);
  model->push_back(obb);
  stride = compute_v12_strides(model);
  obb->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo12OBBImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v12_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[14], outs[17], outs[20]};
  auto* o = model[21]->as<OBBImpl>();
  o->stride = stride;
  return o->forward(det_in);
}

int Yolo12OBBImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto remapped = remap_task_keys(entries, "21");
  return load_state_dict_generic(*this, remapped);
}

// ─── Yolo12ClassifyImpl ───────────────────────────────────────────────────
Yolo12ClassifyImpl::Yolo12ClassifyImpl(Yolo12Scale s, int nc_)
    : scale(s), nc(nc_) {
  // Classify models use BN eps=1e-5 (same convention as v11/v26 cls).
  BnEpsScope eps_scope(1e-5);
  model = register_module("model", torch::nn::ModuleList());
  const auto& y = v12_cls_yaml();
  std::vector<int> ch;
  int c_in = 3;
  for (size_t i = 0; i < y.size(); ++i) {
    const auto& s2 = y[i];
    int prev = (i == 0) ? c_in : ch.back();
    if (s2.kind == "Conv") {
      int c_out = scale_channels_v12(s2.a[0], scale);
      model->push_back(Conv(prev, c_out, s2.a[1], s2.a[2]));
      ch.push_back(c_out);
    } else if (s2.kind == "C3k2") {
      int c_out = scale_channels_v12(s2.a[0], scale);
      int n     = scale_depth_v12(s2.a[1], scale);
      bool c3k  = (s2.a[2] != 0);
      if (scale.width_multiple >= 1.0) c3k = true;
      double e  = (double)s2.a[3] / 100.0;
      model->push_back(C3k2(prev, c_out, n, c3k, e));
      ch.push_back(c_out);
    } else if (s2.kind == "Classify") {
      model->push_back(Classify(prev, nc, /*c_hidden=*/1280));
      ch.push_back(nc);
    }
  }
}

torch::Tensor Yolo12ClassifyImpl::forward(torch::Tensor x) {
  for (size_t i = 0; i < model->size(); ++i) {
    if      (auto m = model[i]->as<ConvImpl>())     x = m->forward(x);
    else if (auto m = model[i]->as<C3k2Impl>())     x = m->forward(x);
    else if (auto m = model[i]->as<ClassifyImpl>()) x = m->forward(x);
  }
  return x;
}

int Yolo12ClassifyImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, entries);
}

}  // namespace yolocpp::models
