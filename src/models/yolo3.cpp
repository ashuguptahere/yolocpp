#include "yolocpp/models/yolo3.hpp"

#include <stdexcept>

namespace yolocpp::models {

namespace {
struct Spec {
  std::vector<int> from;
  int              n_repeat;
  std::string      kind;
  std::vector<int> args;   // Conv: [c_out, k, s]; Bottleneck: [c_out, shortcut(0/1)]
};
const std::vector<Spec>& v3_yaml() {
  // Upstream yolov3.yaml @ depth/width 1.0. Bottleneck shortcut
  // defaults to True; we explicitly mark False with args[1]=0.
  static const std::vector<Spec> y = {
      // Backbone (darknet-53)
      {{-1}, 1, "Conv",       {32, 3, 1}},                  // 0
      {{-1}, 1, "Conv",       {64, 3, 2}},                  // 1 P1/2
      {{-1}, 1, "Bottleneck", {64}},                        // 2
      {{-1}, 1, "Conv",       {128, 3, 2}},                 // 3 P2/4
      {{-1}, 2, "Bottleneck", {128}},                       // 4
      {{-1}, 1, "Conv",       {256, 3, 2}},                 // 5 P3/8
      {{-1}, 8, "Bottleneck", {256}},                       // 6
      {{-1}, 1, "Conv",       {512, 3, 2}},                 // 7 P4/16
      {{-1}, 8, "Bottleneck", {512}},                       // 8
      {{-1}, 1, "Conv",       {1024, 3, 2}},                // 9 P5/32
      {{-1}, 4, "Bottleneck", {1024}},                      // 10
      // Head — P5 branch
      {{-1}, 1, "Bottleneck", {1024, 0}},                   // 11
      {{-1}, 1, "Conv",       {512, 1, 1}},                 // 12
      {{-1}, 1, "Conv",       {1024, 3, 1}},                // 13
      {{-1}, 1, "Conv",       {512, 1, 1}},                 // 14
      {{-1}, 1, "Conv",       {1024, 3, 1}},                // 15  P5
      // Head — P4 branch
      {{-2}, 1, "Conv",       {256, 1, 1}},                 // 16  (from layer 14)
      {{-1}, 1, "Upsample",   {2}},                         // 17
      {{-1, 8}, 1, "Concat",  {}},                          // 18
      {{-1}, 1, "Bottleneck", {512, 0}},                    // 19
      {{-1}, 1, "Bottleneck", {512, 0}},                    // 20
      {{-1}, 1, "Conv",       {256, 1, 1}},                 // 21
      {{-1}, 1, "Conv",       {512, 3, 1}},                 // 22  P4
      // Head — P3 branch
      {{-2}, 1, "Conv",       {128, 1, 1}},                 // 23  (from layer 21)
      {{-1}, 1, "Upsample",   {2}},                         // 24
      {{-1, 6}, 1, "Concat",  {}},                          // 25
      {{-1}, 1, "Bottleneck", {256, 0}},                    // 26
      {{-1}, 2, "Bottleneck", {256, 0}},                    // 27  P3
      {{27, 22, 15}, 1, "Detect", {}},                      // 28
  };
  return y;
}
}  // namespace

Yolo3Impl::Yolo3Impl(Yolo3Scale scale_, int nc_) : scale(scale_), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  const auto& yaml = v3_yaml();
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

    if (s.kind == "Conv") {
      int c_out = s.args[0], k = s.args[1], st = s.args[2];
      model->push_back(Conv(in_ch, c_out, k, st));
      ch.push_back(c_out);
    } else if (s.kind == "Bottleneck") {
      int c_out = s.args[0];
      bool shortcut = !(s.args.size() > 1 && s.args[1] == 0);
      // Upstream parse_model: n=1 → bare module; n>1 → nn.Sequential.
      // Match the keys exactly: bare gives `model.<i>.cv1.*`, Sequential
      // gives `model.<i>.<sub>.cv1.*`.
      if (s.n_repeat == 1) {
        model->push_back(Bottleneck(in_ch, c_out, shortcut, /*g=*/1, /*e=*/0.5));
      } else {
        // Build a Sequential so its children are named "0".."n-1".
        torch::nn::Sequential seq;
        // First Bottleneck takes in_ch (may differ from c_out for the
        // first one in the chain); subsequent take c_out → c_out.
        seq->push_back(Bottleneck(in_ch, c_out, shortcut, 1, 0.5));
        for (int j = 1; j < s.n_repeat; ++j) {
          seq->push_back(Bottleneck(c_out, c_out, shortcut, 1, 0.5));
        }
        model->push_back(seq);
      }
      ch.push_back(c_out);
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
      model->push_back(Detect(nc, det_ch, /*legacy=*/true));
      ch.push_back(0);
    } else {
      throw std::runtime_error("yolo3: unknown layer kind '" + s.kind + "'");
    }
  }
}

torch::Tensor Yolo3Impl::forward_eval(torch::Tensor x) {
  const auto& yaml = v3_yaml();
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

    if (s.kind == "Conv") {
      outs[i] = model[i]->as<ConvImpl>()->forward(in);
    } else if (s.kind == "Bottleneck") {
      if (s.n_repeat == 1) {
        outs[i] = model[i]->as<BottleneckImpl>()->forward(in);
      } else {
        outs[i] = model[i]->as<torch::nn::SequentialImpl>()->forward(in);
      }
    } else if (s.kind == "Upsample") {
      outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    } else if (s.kind == "Concat") {
      outs[i] = in;
    } else if (s.kind == "Detect") {
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

std::vector<torch::Tensor> Yolo3Impl::forward_train(torch::Tensor x) {
  // Mirror of forward_eval up to (but not through) the Detect decode — we
  // return d->forward_features(det_in) directly so V8DetectionLoss can
  // consume the per-scale raw [B, 4*reg_max+nc, H_i, W_i] feature maps.
  const auto& yaml = v3_yaml();
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

    if (s.kind == "Conv") {
      outs[i] = model[i]->as<ConvImpl>()->forward(in);
    } else if (s.kind == "Bottleneck") {
      if (s.n_repeat == 1) {
        outs[i] = model[i]->as<BottleneckImpl>()->forward(in);
      } else {
        outs[i] = model[i]->as<torch::nn::SequentialImpl>()->forward(in);
      }
    } else if (s.kind == "Upsample") {
      outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    } else if (s.kind == "Concat") {
      outs[i] = in;
    } else if (s.kind == "Detect") {
      auto* d = model[i]->as<DetectImpl>();
      std::vector<torch::Tensor> det_in;
      for (int f : s.from) det_in.push_back(outs[f]);
      if (stride.empty()) {
        int img_h = (int)x.size(2);
        for (auto& t : det_in) stride.push_back((double)img_h / (double)t.size(2));
        d->stride = stride;
      }
      return d->forward_features(det_in);
    }
  }
  TORCH_CHECK(false, "Yolo3Impl::forward_train: no Detect layer in yaml");
}

int Yolo3Impl::load_from_state_dict(
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
