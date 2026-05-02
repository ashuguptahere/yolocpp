// RF-DETR projector neck (#65B2). C2f-style 1×1 → 3×3 bottleneck
// stack → 1×1 fusion. Single output level. See header for shape
// reference. Output is at the same spatial grid as the ViT taps
// (no down/up sampling).

#include "yolocpp/models/rfdetr_projector.hpp"

namespace yolocpp::models::rfdetr {

ConvBNImpl::ConvBNImpl(int in_ch, int out_ch, int kernel, int padding)
    : channels_(out_ch) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, kernel)
                            .padding(padding)
                            .bias(false)));
  // Upstream's "bn" is actually a channels-last LayerNorm. Wrap as
  // a sub-module named "bn" so the loaded `<...>.bn.weight` /
  // `bn.bias` dotted paths match.
  auto bn_mod = register_module("bn", std::make_shared<ChannelLastLNImpl>(out_ch));
  weight = bn_mod->weight;
  bias   = bn_mod->bias;
}
torch::Tensor ConvBNImpl::forward(torch::Tensor x) {
  // Apply convolution → channels-last LayerNorm → SiLU.
  // ChannelLastLN does the NCHW↔NHWC permute internally.
  auto bn_mod = std::dynamic_pointer_cast<ChannelLastLNImpl>(
      named_children()["bn"]);
  return torch::silu(bn_mod->forward(conv->forward(x)));
}

ChannelLastLNImpl::ChannelLastLNImpl(int channels) : channels_(channels) {
  weight = register_parameter("weight", torch::ones({channels}));
  bias   = register_parameter("bias",   torch::zeros({channels}));
}
torch::Tensor ChannelLastLNImpl::forward(torch::Tensor x) {
  // [B, C, H, W] → [B, H, W, C] → LN(C) → [B, C, H, W]
  x = x.permute({0, 2, 3, 1});
  x = torch::nn::functional::layer_norm(
      x, torch::nn::functional::LayerNormFuncOptions({channels_})
              .weight(weight).bias(bias).eps(1e-5));
  return x.permute({0, 3, 1, 2});
}

ProjBottleneckImpl::ProjBottleneckImpl(int channels) {
  cv1 = register_module("cv1", ConvBN(channels, channels, /*k=*/3, /*pad=*/1));
  cv2 = register_module("cv2", ConvBN(channels, channels, /*k=*/3, /*pad=*/1));
}
torch::Tensor ProjBottleneckImpl::forward(torch::Tensor x) {
  return cv2->forward(cv1->forward(x));
}

ProjStage0Impl::ProjStage0Impl(int in_ch, int hidden, int n_bottlenecks)
    : hidden_(hidden), n_(n_bottlenecks) {
  // cv1: 1×1 conv from in_ch → hidden.
  cv1 = register_module("cv1", ConvBN(in_ch, hidden, /*k=*/1, /*pad=*/0));
  // m: N bottlenecks, each operating at hidden/2 channels (the
  // upstream C2f convention is cv1 output is split in half along
  // channel dim, then each m block refines the second half).
  m = torch::nn::ModuleList();
  for (int i = 0; i < n_bottlenecks; ++i) {
    m->push_back(ProjBottleneck(hidden / 2));
  }
  register_module("m", m);
  // cv2: 1×1 conv from (hidden + n × hidden/2) → hidden.
  int fanin = hidden + n_bottlenecks * (hidden / 2);
  cv2 = register_module("cv2", ConvBN(fanin, hidden, /*k=*/1, /*pad=*/0));
}
torch::Tensor ProjStage0Impl::forward(torch::Tensor x) {
  auto y0 = cv1->forward(x);
  // Split along channel dim into two halves.
  auto chunks = y0.chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> ys = {chunks[0], chunks[1]};
  for (int i = 0; i < static_cast<int>(m->size()); ++i) {
    ys.push_back(m[i]->as<ProjBottleneckImpl>()->forward(ys.back()));
  }
  return cv2->forward(torch::cat(ys, /*dim=*/1));
}

ProjectorImpl::ProjectorImpl(int n_stages, int tap_concat_ch, int hidden,
                              int n_bottlenecks) {
  stages = torch::nn::ModuleList();
  for (int s = 0; s < n_stages; ++s) {
    // Each stage is a Sequential-like pair: slot 0 = ProjStage0,
    // slot 1 = BatchNorm2d(hidden). We register them via a
    // ModuleList so the dotted path is `stages.<s>.0.*` /
    // `stages.<s>.1.{weight,bias}` matching upstream.
    auto stage_pair = torch::nn::ModuleList();
    int in_ch = (s == 0) ? tap_concat_ch : hidden;
    stage_pair->push_back(ProjStage0(in_ch, hidden, n_bottlenecks));
    // `stages.<i>.1` is upstream's custom channels-last LayerNorm
    // (NOT BatchNorm2d) — same `[hidden]` weight + bias shape, but
    // semantics are 1D LN over the channel dim per spatial cell.
    stage_pair->push_back(ChannelLastLN(hidden));
    stages->push_back(stage_pair);
  }
  register_module("stages", stages);
}

torch::Tensor ProjectorImpl::forward(const std::vector<torch::Tensor>& taps) {
  auto x = torch::cat(taps, /*dim=*/1);
  for (int s = 0; s < static_cast<int>(stages->size()); ++s) {
    auto sp = stages[s]->as<torch::nn::ModuleList>();
    x = (*sp)[0]->as<ProjStage0Impl>()->forward(x);
    x = (*sp)[1]->as<ChannelLastLNImpl>()->forward(x);
  }
  return x;
}

BackboneSlotImpl::BackboneSlotImpl(const Dinov2Cfg& backbone_cfg,
                                     int hidden_dim, int n_proj_stages,
                                     int n_bottlenecks) {
  encoder = register_module("encoder", Dinov2Wrapper(backbone_cfg));
  int tap_concat = 4 * backbone_cfg.hidden_size;
  projector = register_module(
      "projector",
      Projector(n_proj_stages, tap_concat, hidden_dim, n_bottlenecks));
}

torch::Tensor BackboneSlotImpl::forward(torch::Tensor x) {
  auto taps = encoder->forward(std::move(x));
  return projector->forward(taps);
}

}  // namespace yolocpp::models::rfdetr
