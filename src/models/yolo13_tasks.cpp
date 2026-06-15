#include "yolocpp/models/yolo13_tasks.hpp"

#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

namespace {

// ── v13 layer schedule (verbatim from yolo13.cpp kV13Yaml; the Detect at L32 is
//    replaced by a task head appended by each wrapper). ──────────────────────
struct V13Step { std::vector<int> from; std::string kind; };
const std::vector<V13Step> kV13TaskYaml = {
    /* 0  */ {{-1},      "Conv"},   /* 1  */ {{-1},      "Conv"},
    /* 2  */ {{-1},      "DSC3k2"}, /* 3  */ {{-1},      "Conv"},
    /* 4  */ {{-1},      "DSC3k2"}, /* 5  */ {{-1},      "DSConv"},
    /* 6  */ {{-1},      "A2C2f"},  /* 7  */ {{-1},      "DSConv"},
    /* 8  */ {{-1},      "A2C2f"},  /* 9  */ {{4,6,8},   "HyperACE"},
    /* 10 */ {{-1},      "Up"},     /* 11 */ {{9},       "DownsampleConv"},
    /* 12 */ {{6,9},     "FullPADTunnel"}, /* 13 */ {{4,10},    "FullPADTunnel"},
    /* 14 */ {{8,11},    "FullPADTunnel"}, /* 15 */ {{-1},      "Up"},
    /* 16 */ {{-1,12},   "Cat"},    /* 17 */ {{-1},      "DSC3k2"},
    /* 18 */ {{-1,9},    "FullPADTunnel"}, /* 19 */ {{17},      "Up"},
    /* 20 */ {{-1,13},   "Cat"},    /* 21 */ {{-1},      "DSC3k2"},
    /* 22 */ {{10},      "Conv"},   /* 23 */ {{21,22},   "FullPADTunnel"},
    /* 24 */ {{-1},      "Conv"},   /* 25 */ {{-1,18},   "Cat"},
    /* 26 */ {{-1},      "DSC3k2"}, /* 27 */ {{-1,9},    "FullPADTunnel"},
    /* 28 */ {{26},      "Conv"},   /* 29 */ {{-1,14},   "Cat"},
    /* 30 */ {{-1},      "DSC3k2"}, /* 31 */ {{-1,11},   "FullPADTunnel"},
};

inline torch::Tensor resolve_from(int f, const std::vector<torch::Tensor>& outs,
                                  const torch::Tensor& prev) {
  return f == -1 ? prev : outs[f];
}

// Build layers 0..31 of the v13 detect schedule (everything before the Detect
// head). Returns the per-layer output-channel table. Verbatim mirror of
// Yolo13DetectImpl's ctor build (yolo13.cpp), minus the L32 Detect push.
std::vector<int> build_v13_backbone_neck(torch::nn::ModuleList& model,
                                         Yolo13Scale s) {
  int he_base = 8, he = he_base;
  if (s.depth_multiple == kYolo13n.depth_multiple &&
      s.width_multiple == kYolo13n.width_multiple)
    he = (int)(he_base * 0.5);
  else if (s.depth_multiple == kYolo13x.depth_multiple &&
           s.width_multiple == kYolo13x.width_multiple)
    he = (int)(he_base * 1.5);
  bool channel_adjust = !(s.depth_multiple == kYolo13l.depth_multiple &&
                          s.width_multiple == kYolo13l.width_multiple) &&
                        !(s.depth_multiple == kYolo13x.depth_multiple &&
                          s.width_multiple == kYolo13x.width_multiple);
  auto sc = [&](int c) { return scale_channels_v13(c, s); };
  auto sd = [&](int n) { return scale_depth_v13(n, s); };
  bool is_lx = (s.depth_multiple == kYolo13l.depth_multiple ||
                s.depth_multiple == kYolo13x.depth_multiple) &&
               (s.width_multiple == kYolo13l.width_multiple ||
                s.width_multiple == kYolo13x.width_multiple);
  std::vector<int> ch(kV13TaskYaml.size(), 0);

  model->push_back(Conv(3, sc(64), 3, 2));                              ch[0] = sc(64);
  model->push_back(Conv(ch[0], sc(128), 3, 2, 1, 2));                   ch[1] = sc(128);
  model->push_back(DSC3k2(ch[1], sc(256), sd(2), is_lx ? true : false, 0.25, true)); ch[2] = sc(256);
  model->push_back(Conv(ch[2], sc(256), 3, 2, 1, 4));                   ch[3] = sc(256);
  model->push_back(DSC3k2(ch[3], sc(512), sd(2), is_lx ? true : false, 0.25, true)); ch[4] = sc(512);
  bool a2c2f_residual = is_lx;
  double a2c2f_mlp = a2c2f_residual ? 1.5 : 2.0;
  model->push_back(DSConv(ch[4], sc(512), 3, 2));                       ch[5] = sc(512);
  model->push_back(V13A2C2f(ch[5], sc(512), sd(4), true, 4, a2c2f_residual, a2c2f_mlp)); ch[6] = sc(512);
  model->push_back(DSConv(ch[6], sc(1024), 3, 2));                      ch[7] = sc(1024);
  model->push_back(V13A2C2f(ch[7], sc(1024), sd(4), true, 1, a2c2f_residual, a2c2f_mlp)); ch[8] = sc(1024);
  int c2_l9 = sc(512), hyper_n = sd(2);
  model->push_back(HyperACE(ch[6], c2_l9, hyper_n, he, true, true, 0.5, 1.0, "both", channel_adjust)); ch[9] = c2_l9;
  model->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
      .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)));  ch[10] = ch[9];
  bool ds_channel_adjust = channel_adjust;
  model->push_back(DownsampleConv(ch[9], ds_channel_adjust));           ch[11] = ds_channel_adjust ? ch[9] * 2 : ch[9];
  model->push_back(FullPADTunnel());  ch[12] = ch[6];
  model->push_back(FullPADTunnel());  ch[13] = ch[4];
  model->push_back(FullPADTunnel());  ch[14] = ch[8];
  model->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
      .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)));  ch[15] = ch[14];
  ch[16] = ch[15] + ch[12];  model->push_back(torch::nn::Identity());
  model->push_back(DSC3k2(ch[16], sc(512), sd(2), true, 0.5, true));    ch[17] = sc(512);
  model->push_back(FullPADTunnel());  ch[18] = ch[17];
  model->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
      .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest)));  ch[19] = ch[17];
  ch[20] = ch[19] + ch[13];  model->push_back(torch::nn::Identity());
  model->push_back(DSC3k2(ch[20], sc(256), sd(2), true, 0.5, true));    ch[21] = sc(256);
  model->push_back(Conv(ch[10], sc(256), 1, 1));                        ch[22] = sc(256);
  model->push_back(FullPADTunnel());  ch[23] = ch[21];
  model->push_back(Conv(ch[23], sc(256), 3, 2));                        ch[24] = sc(256);
  ch[25] = ch[24] + ch[18];  model->push_back(torch::nn::Identity());
  model->push_back(DSC3k2(ch[25], sc(512), sd(2), true, 0.5, true));    ch[26] = sc(512);
  model->push_back(FullPADTunnel());  ch[27] = ch[26];
  model->push_back(Conv(ch[26], sc(512), 3, 2));                        ch[28] = sc(512);
  ch[29] = ch[28] + ch[14];  model->push_back(torch::nn::Identity());
  model->push_back(DSC3k2(ch[29], sc(1024), sd(2), true, 0.5, true));   ch[30] = sc(1024);
  model->push_back(FullPADTunnel());  ch[31] = ch[30];
  return ch;
}

// Walk layers 0..31, returning every layer's output (mirrors yolo13.cpp's
// forward, stopping before the Detect head).
std::vector<torch::Tensor> forward_v13_backbone_neck(torch::nn::ModuleList& model,
                                                     torch::Tensor x) {
  std::vector<torch::Tensor> outs(kV13TaskYaml.size());
  torch::Tensor prev = x;
  for (size_t i = 0; i < kV13TaskYaml.size(); ++i) {
    const auto& step = kV13TaskYaml[i];
    auto module = model[i];
    torch::Tensor y;
    if (step.kind == "Conv")
      y = module->as<ConvImpl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "DSConv")
      y = module->as<DSConvImpl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "DSC3k2")
      y = module->as<DSC3k2Impl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "A2C2f")
      y = module->as<V13A2C2fImpl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "Up")
      y = module->as<torch::nn::UpsampleImpl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "DownsampleConv")
      y = module->as<DownsampleConvImpl>()->forward(resolve_from(step.from[0], outs, prev));
    else if (step.kind == "HyperACE") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from) ins.push_back(outs[f]);
      y = module->as<HyperACEImpl>()->forward(ins);
    } else if (step.kind == "FullPADTunnel") {
      y = module->as<FullPADTunnelImpl>()->forward(
          resolve_from(step.from[0], outs, prev),
          resolve_from(step.from[1], outs, prev));
    } else if (step.kind == "Cat") {
      std::vector<torch::Tensor> ins;
      for (int f : step.from) ins.push_back(resolve_from(f, outs, prev));
      y = torch::cat(ins, 1);
    } else {
      throw std::runtime_error("yolo13 task: unknown step '" + step.kind + "'");
    }
    outs[i] = y;
    prev = y;
  }
  return outs;
}

std::vector<double> compute_v13_strides(torch::nn::ModuleList& model) {
  torch::NoGradGuard ng;
  auto outs = forward_v13_backbone_neck(model, torch::zeros({1, 3, 256, 256}));
  return {256.0 / (double)outs[23].size(2), 256.0 / (double)outs[27].size(2),
          256.0 / (double)outs[31].size(2)};
}

template <typename M>
int load_state_dict_generic(M& self,
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  std::unordered_map<std::string, at::Tensor> ours;
  for (auto& kv : self.named_parameters()) ours.emplace(kv.key(), kv.value());
  for (auto& kv : self.named_buffers())    ours.emplace(kv.key(), kv.value());
  torch::NoGradGuard ng;
  int copied = 0, skipped = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    if (it->second.sizes() != t.sizes()) { ++skipped; continue; }
    it->second.copy_(t.to(it->second.dtype()).to(it->second.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("v13 task load: copied 0 tensors");
  if (skipped > 0)
    std::cerr << "[v13-task load] skipped " << skipped
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
          sub.rfind("dfl.", 0) == 0)
        nk = head + "detect." + sub;
    }
    out.emplace_back(std::move(nk), t);
  }
  return out;
}

Yolo8Scale v13_shim(Yolo13Scale s) {
  return Yolo8Scale{s.depth_multiple, s.width_multiple, s.max_channels};
}

}  // namespace

// ─── Segment ───────────────────────────────────────────────────────────────
Yolo13SegmentImpl::Yolo13SegmentImpl(Yolo13Scale s, int nc_, int nm,
                                     int npr_unscaled)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v13_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[23], ch[27], ch[31]};
  auto seg = Segment(nc, nm, npr_unscaled, det_ch, v13_shim(scale), false);
  model->push_back(seg);
  stride = compute_v13_strides(model);
  seg->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo13SegmentImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v13_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[23], outs[27], outs[31]};
  auto* seg = model[32]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward(det_in);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
Yolo13SegmentImpl::forward_train_seg(torch::Tensor x) {
  auto outs = forward_v13_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[23], outs[27], outs[31]};
  auto* seg = model[32]->as<SegmentImpl>();
  seg->stride = stride;
  return seg->forward_train(det_in);
}

int Yolo13SegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, remap_task_keys(entries, "32"));
}

// ─── Pose ──────────────────────────────────────────────────────────────────
Yolo13PoseImpl::Yolo13PoseImpl(Yolo13Scale s, int nc_, int num_kpts_, int kpt_dim_)
    : scale(s), nc(nc_), num_kpts(num_kpts_), kpt_dim(kpt_dim_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v13_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[23], ch[27], ch[31]};
  auto pose = Pose(nc, num_kpts, kpt_dim, det_ch, false);
  model->push_back(pose);
  stride = compute_v13_strides(model);
  pose->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo13PoseImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v13_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[23], outs[27], outs[31]};
  auto* p = model[32]->as<PoseImpl>();
  p->stride = stride;
  return p->forward(det_in);
}

int Yolo13PoseImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, remap_task_keys(entries, "32"));
}

// ─── OBB ───────────────────────────────────────────────────────────────────
Yolo13OBBImpl::Yolo13OBBImpl(Yolo13Scale s, int nc_, int ne_)
    : scale(s), nc(nc_), ne(ne_) {
  model = register_module("model", torch::nn::ModuleList());
  auto ch = build_v13_backbone_neck(model, scale);
  std::vector<int> det_ch = {ch[23], ch[27], ch[31]};
  auto obb = OBB(nc, ne, det_ch, false);
  model->push_back(obb);
  stride = compute_v13_strides(model);
  obb->stride = stride;
}

std::tuple<torch::Tensor, torch::Tensor>
Yolo13OBBImpl::forward_eval(torch::Tensor x) {
  auto outs = forward_v13_backbone_neck(model, x);
  std::vector<torch::Tensor> det_in = {outs[23], outs[27], outs[31]};
  auto* o = model[32]->as<OBBImpl>();
  o->stride = stride;
  return o->forward(det_in);
}

int Yolo13OBBImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, remap_task_keys(entries, "32"));
}

// ─── Classify ──────────────────────────────────────────────────────────────
// v13 has no upstream cls; reuse a v13-flavoured backbone (DSConv/DSC3k2 chain)
// ending in the shared Classify head. Trained from scratch.
Yolo13ClassifyImpl::Yolo13ClassifyImpl(Yolo13Scale s, int nc_)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  auto sc = [&](int c) { return scale_channels_v13(c, s); };
  auto sd = [&](int n) { return scale_depth_v13(n, s); };
  int c = 3;
  model->push_back(Conv(c, sc(64), 3, 2));                  c = sc(64);
  model->push_back(Conv(c, sc(128), 3, 2, 1, 2));           c = sc(128);
  model->push_back(DSC3k2(c, sc(256), sd(2), false, 0.25, true)); c = sc(256);
  model->push_back(Conv(c, sc(256), 3, 2, 1, 4));           c = sc(256);
  model->push_back(DSC3k2(c, sc(512), sd(2), false, 0.25, true)); c = sc(512);
  model->push_back(DSConv(c, sc(512), 3, 2));               c = sc(512);
  model->push_back(DSConv(c, sc(1024), 3, 2));              c = sc(1024);
  model->push_back(Classify(c, nc));
}

torch::Tensor Yolo13ClassifyImpl::forward(torch::Tensor x) {
  torch::Tensor y = x;
  for (size_t i = 0; i + 1 < model->size(); ++i) {
    auto m = model[i];
    if (auto* cv = m->as<ConvImpl>())          y = cv->forward(y);
    else if (auto* dc = m->as<DSConvImpl>())   y = dc->forward(y);
    else if (auto* c3 = m->as<DSC3k2Impl>())   y = c3->forward(y);
  }
  return model[model->size() - 1]->as<ClassifyImpl>()->forward(y);
}

int Yolo13ClassifyImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  return load_state_dict_generic(*this, entries);
}

}  // namespace yolocpp::models
