#include "yolocpp/models/yolo3.hpp"

namespace yolocpp::models {

// ─── DarknetResidual ────────────────────────────────────────────────────
DarknetResidualImpl::DarknetResidualImpl(int c) {
  cv1 = register_module("cv1", Conv(c,     c / 2, 1, 1));
  cv2 = register_module("cv2", Conv(c / 2, c,     3, 1));
}
torch::Tensor DarknetResidualImpl::forward(torch::Tensor x) {
  return x + cv2(cv1(x));
}

// ─── DarknetBlock ───────────────────────────────────────────────────────
DarknetBlockImpl::DarknetBlockImpl(int c_in, int c_out, int n) {
  down = register_module("down", Conv(c_in, c_out, 3, 2));
  m    = register_module("m",    torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) m->push_back(DarknetResidual(c_out));
}
torch::Tensor DarknetBlockImpl::forward(torch::Tensor x) {
  x = down(x);
  for (size_t i = 0; i < m->size(); ++i)
    x = m[i]->as<DarknetResidualImpl>()->forward(x);
  return x;
}

// ─── Yolo3 ─────────────────────────────────────────────────────────────
Yolo3Impl::Yolo3Impl(int nc_) : nc(nc_) {
  // Backbone
  stem = register_module("stem", Conv(3, 32, 3, 1));
  b1   = register_module("b1", DarknetBlock(32,  64,  1));
  b2   = register_module("b2", DarknetBlock(64,  128, 2));
  b3   = register_module("b3", DarknetBlock(128, 256, 8));   // P3
  b4   = register_module("b4", DarknetBlock(256, 512, 8));   // P4
  b5   = register_module("b5", DarknetBlock(512, 1024, 4));  // P5

  int out_ch = 3 * (5 + nc);  // 3 anchors × (4 box + 1 obj + nc cls)

  // P5 head — 5 alternating 1×1 / 3×3 convs, then output 1×1.
  p5_pre1 = register_module("p5_pre1", Conv(1024, 512, 1, 1));
  p5_pre2 = register_module("p5_pre2", Conv(512,  1024, 3, 1));
  p5_pre3 = register_module("p5_pre3", Conv(1024, 512, 1, 1));
  p5_pre4 = register_module("p5_pre4", Conv(512,  1024, 3, 1));
  p5_pre5 = register_module("p5_pre5", Conv(1024, 512, 1, 1));
  p5_out_pre = register_module("p5_out_pre", Conv(512, 1024, 3, 1));
  p5_out = register_module("p5_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, out_ch, 1)));

  // P4 head — reduce P5_pre5 (512 ch) → 256 ch, upsample, concat with b4 (512 ch)
  // → 768 ch, then 5-conv tower outputting at 256.
  p4_red  = register_module("p4_red", Conv(512, 256, 1, 1));
  p4_pre1 = register_module("p4_pre1", Conv(768, 256, 1, 1));
  p4_pre2 = register_module("p4_pre2", Conv(256, 512, 3, 1));
  p4_pre3 = register_module("p4_pre3", Conv(512, 256, 1, 1));
  p4_pre4 = register_module("p4_pre4", Conv(256, 512, 3, 1));
  p4_pre5 = register_module("p4_pre5", Conv(512, 256, 1, 1));
  p4_out_pre = register_module("p4_out_pre", Conv(256, 512, 3, 1));
  p4_out = register_module("p4_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(512, out_ch, 1)));

  // P3 head — reduce P4_pre5 (256 ch) → 128 ch, upsample, concat with b3 (256 ch)
  // → 384 ch, then 5-conv tower at 128.
  p3_red  = register_module("p3_red", Conv(256, 128, 1, 1));
  p3_pre1 = register_module("p3_pre1", Conv(384, 128, 1, 1));
  p3_pre2 = register_module("p3_pre2", Conv(128, 256, 3, 1));
  p3_pre3 = register_module("p3_pre3", Conv(256, 128, 1, 1));
  p3_pre4 = register_module("p3_pre4", Conv(128, 256, 3, 1));
  p3_pre5 = register_module("p3_pre5", Conv(256, 128, 1, 1));
  p3_out_pre = register_module("p3_out_pre", Conv(128, 256, 3, 1));
  p3_out = register_module("p3_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, out_ch, 1)));
}

std::vector<torch::Tensor> Yolo3Impl::forward(torch::Tensor x) {
  // Backbone
  auto y = stem(x);
  y = b1(y);
  y = b2(y);
  auto p3_feat = b3(y);
  auto p4_feat = b4(p3_feat);
  auto p5_feat = b5(p4_feat);

  // P5 head
  auto x5 = p5_pre1(p5_feat);
  x5 = p5_pre2(x5);
  x5 = p5_pre3(x5);
  x5 = p5_pre4(x5);
  x5 = p5_pre5(x5);
  auto out5 = p5_out(p5_out_pre(x5));

  // P4 head: upsample(reduce(x5)) ⊕ p4_feat
  auto u4 = torch::nn::functional::interpolate(
      p4_red(x5),
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>{2.0, 2.0})
          .mode(torch::kNearest));
  auto x4 = torch::cat({u4, p4_feat}, /*dim=*/1);
  x4 = p4_pre1(x4);
  x4 = p4_pre2(x4);
  x4 = p4_pre3(x4);
  x4 = p4_pre4(x4);
  x4 = p4_pre5(x4);
  auto out4 = p4_out(p4_out_pre(x4));

  // P3 head: upsample(reduce(x4)) ⊕ p3_feat
  auto u3 = torch::nn::functional::interpolate(
      p3_red(x4),
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>{2.0, 2.0})
          .mode(torch::kNearest));
  auto x3 = torch::cat({u3, p3_feat}, /*dim=*/1);
  x3 = p3_pre1(x3);
  x3 = p3_pre2(x3);
  x3 = p3_pre3(x3);
  x3 = p3_pre4(x3);
  x3 = p3_pre5(x3);
  auto out3 = p3_out(p3_out_pre(x3));

  return {out5, out4, out3};  // strides 32, 16, 8
}

}  // namespace yolocpp::models
