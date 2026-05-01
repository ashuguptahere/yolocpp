#include "yolocpp/models/yolo4.hpp"

namespace yolocpp::models {

// ─── ConvMish ────────────────────────────────────────────────────────────
ConvMishImpl::ConvMishImpl(int c_in, int c_out, int k, int s, int p, int g) {
  if (p < 0) p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).groups(g).bias(false)));
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c_out).eps(1e-4)));
}
torch::Tensor ConvMishImpl::forward(torch::Tensor x) {
  return torch::mish(bn(conv(x)));
}

// ─── ConvLeaky ───────────────────────────────────────────────────────────
ConvLeakyImpl::ConvLeakyImpl(int c_in, int c_out, int k, int s, int p, int g) {
  if (p < 0) p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).groups(g).bias(false)));
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c_out).eps(1e-4)));
}
torch::Tensor ConvLeakyImpl::forward(torch::Tensor x) {
  return torch::leaky_relu(bn(conv(x)), 0.1);
}

// ─── DarknetResidualMish ────────────────────────────────────────────────
DarknetResidualMishImpl::DarknetResidualMishImpl(int c, int hidden) {
  cv1 = register_module("cv1", ConvMish(c,      hidden, 1, 1));
  cv2 = register_module("cv2", ConvMish(hidden, c,      3, 1));
}
torch::Tensor DarknetResidualMishImpl::forward(torch::Tensor x) {
  return x + cv2(cv1(x));
}

// ─── CSPStage ────────────────────────────────────────────────────────────
CSPStageImpl::CSPStageImpl(int c_in, int c_out, int n, bool first_stage) {
  const int c_main   = first_stage ? c_out      : c_out / 2;
  const int hidden   = first_stage ? c_out / 2  : c_out / 2;

  // Registration order matches yolov4.cfg execution order:
  //   down → cv2 (shortcut) → cv1 (main) → m[i] → cv3 → cv4
  // (Darknet's binary streams weights in cfg order with no key names.)
  down = register_module("down", ConvMish(c_in, c_out, 3, 2));
  cv2  = register_module("cv2",  ConvMish(c_out, c_main, 1, 1));
  cv1  = register_module("cv1",  ConvMish(c_out, c_main, 1, 1));
  m    = register_module("m",    torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) m->push_back(DarknetResidualMish(c_main, hidden));
  cv3  = register_module("cv3",  ConvMish(c_main,     c_main, 1, 1));
  cv4  = register_module("cv4",  ConvMish(2 * c_main, c_out,  1, 1));
}
torch::Tensor CSPStageImpl::forward(torch::Tensor x) {
  x = down(x);
  auto y_main = cv1(x);
  for (size_t i = 0; i < m->size(); ++i)
    y_main = m[i]->as<DarknetResidualMishImpl>()->forward(y_main);
  y_main = cv3(y_main);
  auto y_short = cv2(x);
  return cv4(torch::cat({y_main, y_short}, /*dim=*/1));
}

// ─── SPPv4 ───────────────────────────────────────────────────────────────
SPPv4Impl::SPPv4Impl() {
  m5  = register_module("m5",  torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(5).stride(1).padding(2)));
  m9  = register_module("m9",  torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(9).stride(1).padding(4)));
  m13 = register_module("m13", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(13).stride(1).padding(6)));
}
torch::Tensor SPPv4Impl::forward(torch::Tensor x) {
  return torch::cat({m13(x), m9(x), m5(x), x}, /*dim=*/1);
}

// ─── Yolo4 ───────────────────────────────────────────────────────────────
Yolo4Impl::Yolo4Impl(int nc_) : nc(nc_) {
  // Backbone (CSPDarknet53)
  stem = register_module("stem", ConvMish(3, 32, 3, 1));
  s1   = register_module("s1",   CSPStage(32,  64,  1, /*first=*/true));
  s2   = register_module("s2",   CSPStage(64,  128, 2, false));
  s3   = register_module("s3",   CSPStage(128, 256, 8, false));   // P3
  s4   = register_module("s4",   CSPStage(256, 512, 8, false));   // P4
  s5   = register_module("s5",   CSPStage(512, 1024, 4, false));  // P5

  // SPP tower
  spp_pre1 = register_module("spp_pre1", ConvLeaky(1024, 512, 1, 1));
  spp_pre2 = register_module("spp_pre2", ConvLeaky(512, 1024, 3, 1));
  spp_pre3 = register_module("spp_pre3", ConvLeaky(1024, 512, 1, 1));
  spp      = register_module("spp",      SPPv4());
  spp_post1 = register_module("spp_post1", ConvLeaky(2048, 512, 1, 1));
  spp_post2 = register_module("spp_post2", ConvLeaky(512, 1024, 3, 1));
  spp_post3 = register_module("spp_post3", ConvLeaky(1024, 512, 1, 1));

  // Top-down P5 → P4
  p4_td_red  = register_module("p4_td_red",  ConvLeaky(512, 256, 1, 1));
  p4_lateral = register_module("p4_lateral", ConvLeaky(512, 256, 1, 1));
  p4_td1 = register_module("p4_td1", ConvLeaky(512, 256, 1, 1));
  p4_td2 = register_module("p4_td2", ConvLeaky(256, 512, 3, 1));
  p4_td3 = register_module("p4_td3", ConvLeaky(512, 256, 1, 1));
  p4_td4 = register_module("p4_td4", ConvLeaky(256, 512, 3, 1));
  p4_td5 = register_module("p4_td5", ConvLeaky(512, 256, 1, 1));

  // Top-down P4 → P3
  p3_td_red  = register_module("p3_td_red",  ConvLeaky(256, 128, 1, 1));
  p3_lateral = register_module("p3_lateral", ConvLeaky(256, 128, 1, 1));
  p3_td1 = register_module("p3_td1", ConvLeaky(256, 128, 1, 1));
  p3_td2 = register_module("p3_td2", ConvLeaky(128, 256, 3, 1));
  p3_td3 = register_module("p3_td3", ConvLeaky(256, 128, 1, 1));
  p3_td4 = register_module("p3_td4", ConvLeaky(128, 256, 3, 1));
  p3_td5 = register_module("p3_td5", ConvLeaky(256, 128, 1, 1));

  // P3 head (output) — emitted right after top-down P3 in cfg order.
  const int out_ch = 3 * (5 + nc);
  p3_out_pre = register_module("p3_out_pre", ConvLeaky(128, 256, 3, 1));
  p3_out = register_module("p3_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, out_ch, 1)));

  // Bottom-up P3 → P4 → P4 head
  p4_bu_down = register_module("p4_bu_down", ConvLeaky(128, 256, 3, 2));
  p4_bu1 = register_module("p4_bu1", ConvLeaky(512, 256, 1, 1));
  p4_bu2 = register_module("p4_bu2", ConvLeaky(256, 512, 3, 1));
  p4_bu3 = register_module("p4_bu3", ConvLeaky(512, 256, 1, 1));
  p4_bu4 = register_module("p4_bu4", ConvLeaky(256, 512, 3, 1));
  p4_bu5 = register_module("p4_bu5", ConvLeaky(512, 256, 1, 1));
  p4_out_pre = register_module("p4_out_pre", ConvLeaky(256, 512, 3, 1));
  p4_out = register_module("p4_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(512, out_ch, 1)));

  // Bottom-up P4 → P5 → P5 head
  p5_bu_down = register_module("p5_bu_down", ConvLeaky(256, 512, 3, 2));
  p5_bu1 = register_module("p5_bu1", ConvLeaky(1024, 512, 1, 1));
  p5_bu2 = register_module("p5_bu2", ConvLeaky(512, 1024, 3, 1));
  p5_bu3 = register_module("p5_bu3", ConvLeaky(1024, 512, 1, 1));
  p5_bu4 = register_module("p5_bu4", ConvLeaky(512, 1024, 3, 1));
  p5_bu5 = register_module("p5_bu5", ConvLeaky(1024, 512, 1, 1));
  p5_out_pre = register_module("p5_out_pre", ConvLeaky(512, 1024, 3, 1));
  p5_out = register_module("p5_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, out_ch, 1)));
}

std::vector<torch::Tensor> Yolo4Impl::forward(torch::Tensor x) {
  // Backbone
  auto y  = stem(x);
  y       = s1(y);
  y       = s2(y);
  auto p3 = s3(y);   // 256 ch, stride 8
  auto p4 = s4(p3);  // 512 ch, stride 16
  auto p5 = s5(p4);  // 1024 ch, stride 32

  // SPP tower
  auto x5 = spp_pre3(spp_pre2(spp_pre1(p5)));
  x5      = spp(x5);
  x5      = spp_post3(spp_post2(spp_post1(x5)));   // 512 ch

  // Top-down P5 → P4
  auto u4 = torch::nn::functional::interpolate(
      p4_td_red(x5),
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest));
  auto x4 = torch::cat({p4_lateral(p4), u4}, /*dim=*/1);  // 512
  x4 = p4_td1(x4);
  x4 = p4_td2(x4);
  x4 = p4_td3(x4);
  x4 = p4_td4(x4);
  x4 = p4_td5(x4);                                        // 256

  // Top-down P4 → P3
  auto u3 = torch::nn::functional::interpolate(
      p3_td_red(x4),
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>{2.0, 2.0}).mode(torch::kNearest));
  auto x3 = torch::cat({p3_lateral(p3), u3}, /*dim=*/1);  // 256
  x3 = p3_td1(x3);
  x3 = p3_td2(x3);
  x3 = p3_td3(x3);
  x3 = p3_td4(x3);
  x3 = p3_td5(x3);                                        // 128

  // P3 output
  auto out3 = p3_out(p3_out_pre(x3));

  // Bottom-up P3 → P4
  auto d4 = p4_bu_down(x3);                               // 256
  auto y4 = torch::cat({d4, x4}, /*dim=*/1);              // 512
  y4 = p4_bu1(y4);
  y4 = p4_bu2(y4);
  y4 = p4_bu3(y4);
  y4 = p4_bu4(y4);
  y4 = p4_bu5(y4);                                        // 256
  auto out4 = p4_out(p4_out_pre(y4));

  // Bottom-up P4 → P5
  auto d5 = p5_bu_down(y4);                               // 512
  auto y5 = torch::cat({d5, x5}, /*dim=*/1);              // 1024
  y5 = p5_bu1(y5);
  y5 = p5_bu2(y5);
  y5 = p5_bu3(y5);
  y5 = p5_bu4(y5);
  y5 = p5_bu5(y5);                                        // 512
  auto out5 = p5_out(p5_out_pre(y5));

  return {out5, out4, out3};  // strides 32, 16, 8
}

std::vector<torch::Tensor> Yolo4Impl::forward_train(torch::Tensor x) {
  // Reorder to stride-ASCENDING (P3, P4, P5) for V7DetectionLoss.
  auto raw = forward(x);
  if (stride.empty()) {
    int img_h = (int)x.size(2);
    // raw is in {P5, P4, P3} order; compute strides accordingly.
    std::vector<double> str_desc;
    for (auto& t : raw) str_desc.push_back((double)img_h / (double)t.size(2));
    stride = {str_desc[2], str_desc[1], str_desc[0]};
  }
  return {raw[2], raw[1], raw[0]};   // P3, P4, P5
}

// ─── forward_eval — anchor decode, returns [B, 4 + nc, A] ────────────────
//
// yolov4.cfg anchors (in pixel units on 608² input — Darknet calibrates
// them to the cfg's `width`/`height`, NOT to feature units):
//   P3 (mask 0,1,2): (12,16) (19,36)  (40,28)
//   P4 (mask 3,4,5): (36,75) (76,55)  (72,146)
//   P5 (mask 6,7,8): (142,110) (192,243) (459,401)
//
// scale_x_y per scale (from yolov4.cfg [yolo] sections):
//   P3: 1.2   P4: 1.1   P5: 1.05
//
// xy decode (modern AlexeyAB form, with scale_x_y bias-fix):
//   bx_cell = sigmoid(tx) * s - 0.5*(s - 1) + gx     (cell-relative units)
//   bx_pix  = bx_cell * stride                        (input-image pixels)
// wh decode:
//   bw_pix  = anchor_w_in_pixels * exp(tw)            (input-image pixels)
//
// Final score = sigmoid(obj) * sigmoid(cls). NMS will threshold on this.
namespace {
struct ScaleSpec {
  int    stride;
  float  scale_xy;
  std::array<std::pair<float, float>, 3> anchors;  // (w, h) in pixels @ 608
};
const ScaleSpec kV4Scales[3] = {
    // P5 (out index 0 from forward()) — stride 32, anchors 6/7/8
    {32, 1.05f, {{ {142.f, 110.f}, {192.f, 243.f}, {459.f, 401.f} }}},
    // P4 (index 1) — stride 16, anchors 3/4/5
    {16, 1.10f, {{ { 36.f,  75.f}, { 76.f,  55.f}, { 72.f, 146.f} }}},
    // P3 (index 2) — stride  8, anchors 0/1/2
    { 8, 1.20f, {{ { 12.f,  16.f}, { 19.f,  36.f}, { 40.f,  28.f} }}},
};
constexpr int kV4CalibrationSize = 608;
}  // namespace

torch::Tensor Yolo4Impl::forward_eval(torch::Tensor x) {
  auto raw = forward(x);                          // {P5, P4, P3} raw logits
  const int B   = (int)x.size(0);
  const int H_in = (int)x.size(2);
  const int nc_ = nc;

  std::vector<torch::Tensor> per_scale;
  per_scale.reserve(3);
  auto dev = x.device();

  for (int si = 0; si < 3; ++si) {
    const auto& sp = kV4Scales[si];
    auto t = raw[si];                              // [B, 3*(5+nc), Hf, Wf]
    const int Hf = (int)t.size(2);
    const int Wf = (int)t.size(3);
    // Reshape: [B, 3, 5+nc, Hf, Wf] → permute to [B, 3, Hf, Wf, 5+nc]
    t = t.view({B, 3, 5 + nc_, Hf, Wf}).permute({0, 1, 3, 4, 2}).contiguous();

    auto tx = t.index({"...", 0});
    auto ty = t.index({"...", 1});
    auto tw = t.index({"...", 2});
    auto th = t.index({"...", 3});
    auto to = t.index({"...", 4});
    auto tc = t.index({"...", torch::indexing::Slice(5, 5 + nc_)});

    // Cell grid
    auto gy = torch::arange(Hf, torch::TensorOptions().device(dev).dtype(t.dtype()))
                  .view({1, 1, Hf, 1});
    auto gx = torch::arange(Wf, torch::TensorOptions().device(dev).dtype(t.dtype()))
                  .view({1, 1, 1, Wf});

    // Anchor tensors per head (in pixel units rescaled to current input).
    // yolov4.cfg anchors are calibrated for 608² input; rescale to current
    // imgsz so the same .pt works at any letterbox size.
    const float img_scale = (float)H_in / (float)kV4CalibrationSize;
    auto aw = torch::tensor({sp.anchors[0].first  * img_scale,
                             sp.anchors[1].first  * img_scale,
                             sp.anchors[2].first  * img_scale},
                            torch::TensorOptions().device(dev).dtype(t.dtype()))
                  .view({1, 3, 1, 1});
    auto ah = torch::tensor({sp.anchors[0].second * img_scale,
                             sp.anchors[1].second * img_scale,
                             sp.anchors[2].second * img_scale},
                            torch::TensorOptions().device(dev).dtype(t.dtype()))
                  .view({1, 3, 1, 1});

    auto sxy = sp.scale_xy;
    auto bx = (torch::sigmoid(tx) * sxy - 0.5f * (sxy - 1.0f) + gx) * (float)sp.stride;
    auto by = (torch::sigmoid(ty) * sxy - 0.5f * (sxy - 1.0f) + gy) * (float)sp.stride;
    auto bw = aw * torch::exp(tw);
    auto bh = ah * torch::exp(th);

    auto x1 = bx - 0.5f * bw;
    auto y1 = by - 0.5f * bh;
    auto x2 = bx + 0.5f * bw;
    auto y2 = by + 0.5f * bh;

    auto obj = torch::sigmoid(to);                                  // [B, 3, Hf, Wf]
    auto cls = torch::sigmoid(tc);                                  // [B, 3, Hf, Wf, nc]
    auto score = cls * obj.unsqueeze(-1);                           // [B, 3, Hf, Wf, nc]

    // Stack into [B, 3, Hf, Wf, 4 + nc]
    auto box = torch::stack({x1, y1, x2, y2}, /*dim=*/-1);          // [B, 3, Hf, Wf, 4]
    auto packed = torch::cat({box, score}, /*dim=*/-1);             // [B, 3, Hf, Wf, 4+nc]

    // Flatten anchors+grid → A_i = 3 * Hf * Wf, output [B, A_i, 4+nc]
    packed = packed.view({B, 3 * Hf * Wf, 4 + nc_});
    per_scale.push_back(packed);
  }

  // Concat over A → [B, A, 4+nc] then permute to [B, 4+nc, A] for NMS.
  auto out = torch::cat(per_scale, /*dim=*/1);
  return out.permute({0, 2, 1}).contiguous();
}

// ─── load_from_state_dict — match by name ────────────────────────────────
int Yolo4Impl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params  = this->named_parameters(/*recurse=*/true);
  auto buffers = this->named_buffers(/*recurse=*/true);
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
