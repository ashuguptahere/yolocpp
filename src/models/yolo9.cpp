#include "yolocpp/models/yolo9.hpp"

#include <stdexcept>

namespace yolocpp::models {

// ─── Yolo9RepConv ────────────────────────────────────────────────────────
Yolo9RepConvImpl::Yolo9RepConvImpl(int c_in, int c_out) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, 3)
                            .stride(1).padding(1).bias(true)));
}
torch::Tensor Yolo9RepConvImpl::forward(torch::Tensor x) {
  return torch::silu(conv(x));
}

// ─── Yolo9RepBottleneck ──────────────────────────────────────────────────
Yolo9RepBottleneckImpl::Yolo9RepBottleneckImpl(int c1, int c2) {
  cv1 = register_module("cv1", Yolo9RepConv(c1, c2));
  cv2 = register_module("cv2", Conv(c2, c2, 3, 1));
  add = (c1 == c2);
}
torch::Tensor Yolo9RepBottleneckImpl::forward(torch::Tensor x) {
  auto y = cv2(cv1(x));
  return add ? (x + y) : y;
}

// ─── RepCSP ──────────────────────────────────────────────────────────────
RepCSPImpl::RepCSPImpl(int c1, int c2, int n) {
  const int c_ = c2 / 2;
  cv1 = register_module("cv1", Conv(c1, c_, 1, 1));
  cv2 = register_module("cv2", Conv(c1, c_, 1, 1));
  cv3 = register_module("cv3", Conv(2 * c_, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) m->push_back(Yolo9RepBottleneck(c_, c_));
}
torch::Tensor RepCSPImpl::forward(torch::Tensor x) {
  auto a = cv1(x);
  for (size_t i = 0; i < m->size(); ++i)
    a = m[i]->as<Yolo9RepBottleneckImpl>()->forward(a);
  auto b = cv2(x);
  return cv3(torch::cat({a, b}, /*dim=*/1));
}

// ─── RepNCSPELAN4 ────────────────────────────────────────────────────────
RepNCSPELAN4Impl::RepNCSPELAN4Impl(int c1, int c2, int c3, int c4, int n) {
  c_split = c3 / 2;
  cv1 = register_module("cv1", Conv(c1, c3, 1, 1));
  cv2 = register_module("cv2", torch::nn::ModuleList());
  cv2->push_back(RepCSP(c3 / 2, c4, n));
  cv2->push_back(Conv(c4, c4, 3, 1));
  cv3 = register_module("cv3", torch::nn::ModuleList());
  cv3->push_back(RepCSP(c4, c4, n));
  cv3->push_back(Conv(c4, c4, 3, 1));
  cv4 = register_module("cv4", Conv(c3 + 2 * c4, c2, 1, 1));
}
torch::Tensor RepNCSPELAN4Impl::forward(torch::Tensor x) {
  auto y      = cv1(x);
  auto chunks = y.chunk(2, /*dim=*/1);
  auto y2     = cv2[0]->as<RepCSPImpl>()->forward(chunks[1]);
  y2          = cv2[1]->as<ConvImpl>()->forward(y2);
  auto y3     = cv3[0]->as<RepCSPImpl>()->forward(y2);
  y3          = cv3[1]->as<ConvImpl>()->forward(y3);
  return cv4(torch::cat({chunks[0], chunks[1], y2, y3}, /*dim=*/1));
}

// ─── ADown ───────────────────────────────────────────────────────────────
ADownImpl::ADownImpl(int c1, int c2) {
  c   = c2 / 2;
  cv1 = register_module("cv1", Conv(c1 / 2, c, 3, 2, 1));
  cv2 = register_module("cv2", Conv(c1 / 2, c, 1, 1, 0));
}
torch::Tensor ADownImpl::forward(torch::Tensor x) {
  x = torch::nn::functional::avg_pool2d(
      x, torch::nn::functional::AvgPool2dFuncOptions(2).stride(1).padding(0));
  auto chunks = x.chunk(2, /*dim=*/1);
  auto x1 = cv1(chunks[0]);
  auto x2 = torch::nn::functional::max_pool2d(
      chunks[1],
      torch::nn::functional::MaxPool2dFuncOptions(3).stride(2).padding(1));
  x2 = cv2(x2);
  return torch::cat({x1, x2}, /*dim=*/1);
}

// ─── AConv ──────────────────────────────────────────────────────────────
//
// AConv(c1, c2):
//   cv1 = Conv(c1, c2, 3, 2, 1)
//
// forward: cv1(avg_pool2d(x, 2, 1, 0))
AConvImpl::AConvImpl(int c1, int c2) {
  cv1 = register_module("cv1", Conv(c1, c2, 3, 2, 1));
}
torch::Tensor AConvImpl::forward(torch::Tensor x) {
  x = torch::nn::functional::avg_pool2d(
      x, torch::nn::functional::AvgPool2dFuncOptions(2).stride(1).padding(0));
  return cv1(x);
}

// ─── ELAN1 ──────────────────────────────────────────────────────────────
//
// ELAN1(c1, c2, c3, c4):
//   cv1 = Conv(c1, c3, 1, 1)
//   cv2 = Conv(c3//2, c4, 3, 1)
//   cv3 = Conv(c4, c4, 3, 1)
//   cv4 = Conv(c3 + 2*c4, c2, 1, 1)
//
// forward: same as RepNCSPELAN4 but the inner lanes are bare Convs.
ELAN1Impl::ELAN1Impl(int c1, int c2, int c3, int c4) {
  c_split = c3 / 2;
  cv1 = register_module("cv1", Conv(c1, c3, 1, 1));
  cv2 = register_module("cv2", Conv(c3 / 2, c4, 3, 1));
  cv3 = register_module("cv3", Conv(c4, c4, 3, 1));
  cv4 = register_module("cv4", Conv(c3 + 2 * c4, c2, 1, 1));
}
torch::Tensor ELAN1Impl::forward(torch::Tensor x) {
  auto y      = cv1(x);
  auto chunks = y.chunk(2, /*dim=*/1);
  auto y2     = cv2(chunks[1]);
  auto y3     = cv3(y2);
  return cv4(torch::cat({chunks[0], chunks[1], y2, y3}, /*dim=*/1));
}

// ─── SPPELAN ─────────────────────────────────────────────────────────────
SPPELANImpl::SPPELANImpl(int c1, int c2, int c3, int k) {
  cv1 = register_module("cv1", Conv(c1, c3, 1, 1));
  cv2 = register_module("cv2", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
  cv3 = register_module("cv3", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
  cv4 = register_module("cv4", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
  cv5 = register_module("cv5", Conv(4 * c3, c2, 1, 1));
}
torch::Tensor SPPELANImpl::forward(torch::Tensor x) {
  auto y0 = cv1(x);
  auto y1 = cv2(y0);
  auto y2 = cv3(y1);
  auto y3 = cv4(y2);
  return cv5(torch::cat({y0, y1, y2, y3}, /*dim=*/1));
}

// ─── CBLinear ────────────────────────────────────────────────────────────
CBLinearImpl::CBLinearImpl(int c1, std::vector<int> c2s_list) : c2s(std::move(c2s_list)) {
  int sum_c = 0;
  for (int c : c2s) sum_c += c;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, sum_c, 1)
                            .stride(1).padding(0).bias(true)));
}
torch::Tensor CBLinearImpl::forward(torch::Tensor x) {
  // Returns the FULL concatenated output [B, sum(c2s), H, W]. Downstream
  // CBFuse will slice by branch.
  return conv(x);
}

// ─── CBFuse ──────────────────────────────────────────────────────────────
CBFuseImpl::CBFuseImpl(std::vector<int> idx_,
                        std::vector<std::vector<int>> c2s_per_input_)
    : idx(std::move(idx_)), c2s_per_input(std::move(c2s_per_input_)) {}
torch::Tensor CBFuseImpl::forward(const std::vector<torch::Tensor>& xs) {
  // xs: [cb_0, cb_1, ..., cb_{n-1}, current]. Last is the anchor (target
  // spatial). For each i in 0..n-1, slice cb_i by idx[i] using
  // c2s_per_input[i] (start = sum(c2s[0..idx-1]), len = c2s[idx]).
  TORCH_CHECK(xs.size() == idx.size() + 1, "CBFuse: input count mismatch");
  auto target = xs.back();
  int H = (int)target.size(2);
  int W = (int)target.size(3);
  auto sum = target;
  for (size_t i = 0; i + 1 < xs.size(); ++i) {
    int branch_idx = idx[i];
    int start = 0;
    for (int k = 0; k < branch_idx; ++k) start += c2s_per_input[i][k];
    int len   = c2s_per_input[i][branch_idx];
    auto sliced = xs[i].narrow(/*dim=*/1, /*start=*/start, /*length=*/len);
    if ((int)sliced.size(2) != H || (int)sliced.size(3) != W) {
      sliced = torch::nn::functional::interpolate(
          sliced,
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{H, W})
              .mode(torch::kNearest));
    }
    sum = sum + sliced;
  }
  return sum;
}

// ─── yolov9c yaml-walker ─────────────────────────────────────────────────
namespace {
struct Spec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> args;
};
const std::vector<Spec>& v9c_yaml() {
  static const std::vector<Spec> y = {
      {{-1}, "Conv",         {64, 3, 2}},                       // 0
      {{-1}, "Conv",         {128, 3, 2}},                      // 1
      {{-1}, "RepNCSPELAN4", {256, 128, 64, 1}},                // 2
      {{-1}, "ADown",        {256}},                            // 3
      {{-1}, "RepNCSPELAN4", {512, 256, 128, 1}},               // 4
      {{-1}, "ADown",        {512}},                            // 5
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 1}},               // 6
      {{-1}, "ADown",        {512}},                            // 7
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 1}},               // 8
      {{-1}, "SPPELAN",      {512, 256}},                       // 9
      {{-1}, "Upsample",     {2}},                              // 10
      {{-1, 6}, "Concat",    {}},                               // 11
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 1}},               // 12
      {{-1}, "Upsample",     {2}},                              // 13
      {{-1, 4}, "Concat",    {}},                               // 14
      {{-1}, "RepNCSPELAN4", {256, 256, 128, 1}},               // 15
      {{-1}, "ADown",        {256}},                            // 16
      {{-1, 12}, "Concat",   {}},                               // 17
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 1}},               // 18
      {{-1}, "ADown",        {512}},                            // 19
      {{-1, 9}, "Concat",    {}},                               // 20
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 1}},               // 21
      {{15, 18, 21}, "Detect", {}},                             // 22
  };
  return y;
}
const std::vector<Spec>& v9t_yaml() {
  static const std::vector<Spec> y = {
      {{-1}, "Conv",         {16, 3, 2}},                       // 0
      {{-1}, "Conv",         {32, 3, 2}},                       // 1
      {{-1}, "ELAN1",        {32, 32, 16}},                     // 2
      {{-1}, "AConv",        {64}},                             // 3
      {{-1}, "RepNCSPELAN4", {64, 64, 32, 3}},                  // 4
      {{-1}, "AConv",        {96}},                             // 5
      {{-1}, "RepNCSPELAN4", {96, 96, 48, 3}},                  // 6
      {{-1}, "AConv",        {128}},                            // 7
      {{-1}, "RepNCSPELAN4", {128, 128, 64, 3}},                // 8
      {{-1}, "SPPELAN",      {128, 64}},                        // 9
      {{-1}, "Upsample",     {2}},                              // 10
      {{-1, 6}, "Concat",    {}},                               // 11
      {{-1}, "RepNCSPELAN4", {96, 96, 48, 3}},                  // 12
      {{-1}, "Upsample",     {2}},                              // 13
      {{-1, 4}, "Concat",    {}},                               // 14
      {{-1}, "RepNCSPELAN4", {64, 64, 32, 3}},                  // 15
      {{-1}, "AConv",        {48}},                             // 16
      {{-1, 12}, "Concat",   {}},                               // 17
      {{-1}, "RepNCSPELAN4", {96, 96, 48, 3}},                  // 18
      {{-1}, "AConv",        {64}},                             // 19
      {{-1, 9}, "Concat",    {}},                               // 20
      {{-1}, "RepNCSPELAN4", {128, 128, 64, 3}},                // 21
      {{15, 18, 21}, "Detect", {}},                             // 22
  };
  return y;
}
const std::vector<Spec>& v9s_yaml() {
  static const std::vector<Spec> y = {
      {{-1}, "Conv",         {32, 3, 2}},                       // 0
      {{-1}, "Conv",         {64, 3, 2}},                       // 1
      {{-1}, "ELAN1",        {64, 64, 32}},                     // 2
      {{-1}, "AConv",        {128}},                            // 3
      {{-1}, "RepNCSPELAN4", {128, 128, 64, 3}},                // 4
      {{-1}, "AConv",        {192}},                            // 5
      {{-1}, "RepNCSPELAN4", {192, 192, 96, 3}},                // 6
      {{-1}, "AConv",        {256}},                            // 7
      {{-1}, "RepNCSPELAN4", {256, 256, 128, 3}},               // 8
      {{-1}, "SPPELAN",      {256, 128}},                       // 9
      {{-1}, "Upsample",     {2}},                              // 10
      {{-1, 6}, "Concat",    {}},                               // 11
      {{-1}, "RepNCSPELAN4", {192, 192, 96, 3}},                // 12
      {{-1}, "Upsample",     {2}},                              // 13
      {{-1, 4}, "Concat",    {}},                               // 14
      {{-1}, "RepNCSPELAN4", {128, 128, 64, 3}},                // 15
      {{-1}, "AConv",        {96}},                             // 16
      {{-1, 12}, "Concat",   {}},                               // 17
      {{-1}, "RepNCSPELAN4", {192, 192, 96, 3}},                // 18
      {{-1}, "AConv",        {128}},                            // 19
      {{-1, 9}, "Concat",    {}},                               // 20
      {{-1}, "RepNCSPELAN4", {256, 256, 128, 3}},               // 21
      {{15, 18, 21}, "Detect", {}},                             // 22
  };
  return y;
}
const std::vector<Spec>& v9m_yaml() {
  static const std::vector<Spec> y = {
      {{-1}, "Conv",         {32, 3, 2}},                       // 0
      {{-1}, "Conv",         {64, 3, 2}},                       // 1
      {{-1}, "RepNCSPELAN4", {128, 128, 64, 1}},                // 2
      {{-1}, "AConv",        {240}},                            // 3
      {{-1}, "RepNCSPELAN4", {240, 240, 120, 1}},               // 4
      {{-1}, "AConv",        {360}},                            // 5
      {{-1}, "RepNCSPELAN4", {360, 360, 180, 1}},               // 6
      {{-1}, "AConv",        {480}},                            // 7
      {{-1}, "RepNCSPELAN4", {480, 480, 240, 1}},               // 8
      {{-1}, "SPPELAN",      {480, 240}},                       // 9
      {{-1}, "Upsample",     {2}},                              // 10
      {{-1, 6}, "Concat",    {}},                               // 11
      {{-1}, "RepNCSPELAN4", {360, 360, 180, 1}},               // 12
      {{-1}, "Upsample",     {2}},                              // 13
      {{-1, 4}, "Concat",    {}},                               // 14
      {{-1}, "RepNCSPELAN4", {240, 240, 120, 1}},               // 15
      {{-1}, "AConv",        {180}},                            // 16
      {{-1, 12}, "Concat",   {}},                               // 17
      {{-1}, "RepNCSPELAN4", {360, 360, 180, 1}},               // 18
      {{-1}, "AConv",        {240}},                            // 19
      {{-1, 9}, "Concat",    {}},                               // 20
      {{-1}, "RepNCSPELAN4", {480, 480, 240, 1}},               // 21
      {{15, 18, 21}, "Detect", {}},                             // 22
  };
  return y;
}
const std::vector<Spec>& v9e_yaml() {
  // yolov9e: two-pass backbone with CBLinear/CBFuse multi-level
  // connectivity. The "primary" path is layers 0-9 (mirrors v9c).
  // Layers 10-14 are CBLinear taps that project select primary outputs
  // into multi-branch channel splits. Layers 15-29 form the secondary
  // path that re-ingests the input image and pulls in matching branches
  // via CBFuse at each downsample step. The neck/head (30-42) attach to
  // the secondary path.
  // Args convention: for CBLinear, args is the c2s list. For CBFuse,
  // args is the idx list (length = len(from) - 1).
  static const std::vector<Spec> y = {
      {{-1}, "Identity", {}},                                // 0  raw input
      {{-1}, "Conv",     {64, 3, 2}},                         // 1  P1/2
      {{-1}, "Conv",     {128, 3, 2}},                        // 2  P2/4
      {{-1}, "RepNCSPELAN4", {256, 128, 64, 2}},              // 3
      {{-1}, "ADown",    {256}},                              // 4  P3/8
      {{-1}, "RepNCSPELAN4", {512, 256, 128, 2}},             // 5
      {{-1}, "ADown",    {512}},                              // 6  P4/16
      {{-1}, "RepNCSPELAN4", {1024, 512, 256, 2}},            // 7
      {{-1}, "ADown",    {1024}},                             // 8  P5/32
      {{-1}, "RepNCSPELAN4", {1024, 512, 256, 2}},            // 9
      // CBLinear taps from primary
      {{1}, "CBLinear", {64}},                                // 10
      {{3}, "CBLinear", {64, 128}},                           // 11
      {{5}, "CBLinear", {64, 128, 256}},                      // 12
      {{7}, "CBLinear", {64, 128, 256, 512}},                 // 13
      {{9}, "CBLinear", {64, 128, 256, 512, 1024}},           // 14
      // Secondary backbone (re-ingests image at layer 0)
      {{0}, "Conv",     {64, 3, 2}},                          // 15  P1/2
      {{10, 11, 12, 13, 14, -1}, "CBFuse", {0, 0, 0, 0, 0}},  // 16
      {{-1}, "Conv",     {128, 3, 2}},                        // 17  P2/4
      {{11, 12, 13, 14, -1}, "CBFuse", {1, 1, 1, 1}},         // 18
      {{-1}, "RepNCSPELAN4", {256, 128, 64, 2}},              // 19
      {{-1}, "ADown",    {256}},                              // 20  P3/8
      {{12, 13, 14, -1}, "CBFuse", {2, 2, 2}},                // 21
      {{-1}, "RepNCSPELAN4", {512, 256, 128, 2}},             // 22
      {{-1}, "ADown",    {512}},                              // 23  P4/16
      {{13, 14, -1}, "CBFuse", {3, 3}},                       // 24
      {{-1}, "RepNCSPELAN4", {1024, 512, 256, 2}},            // 25
      {{-1}, "ADown",    {1024}},                             // 26  P5/32
      {{14, -1}, "CBFuse", {4}},                              // 27
      {{-1}, "RepNCSPELAN4", {1024, 512, 256, 2}},            // 28
      {{-1}, "SPPELAN",  {512, 256}},                         // 29
      // Head
      {{-1}, "Upsample", {2}},                                // 30
      {{-1, 25}, "Concat", {}},                               // 31
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 2}},             // 32
      {{-1}, "Upsample", {2}},                                // 33
      {{-1, 22}, "Concat", {}},                               // 34
      {{-1}, "RepNCSPELAN4", {256, 256, 128, 2}},             // 35  P3 head
      {{-1}, "ADown",    {256}},                              // 36
      {{-1, 32}, "Concat", {}},                               // 37
      {{-1}, "RepNCSPELAN4", {512, 512, 256, 2}},             // 38  P4 head
      {{-1}, "ADown",    {512}},                              // 39
      {{-1, 29}, "Concat", {}},                               // 40
      {{-1}, "RepNCSPELAN4", {512, 1024, 512, 2}},            // 41  P5 head
      {{35, 38, 41}, "Detect", {}},                           // 42
  };
  return y;
}
const std::vector<Spec>& v9_yaml_for(yolocpp::models::Yolo9Scale s) {
  using yolocpp::models::Yolo9Scale;
  switch (s) {
    case Yolo9Scale::T: return v9t_yaml();
    case Yolo9Scale::S: return v9s_yaml();
    case Yolo9Scale::M: return v9m_yaml();
    case Yolo9Scale::E: return v9e_yaml();
    case Yolo9Scale::C:
    default:            return v9c_yaml();
  }
}
}  // namespace

Yolo9Impl::Yolo9Impl(Yolo9Scale scale_, int nc_) : scale(scale_), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  const auto& yaml = v9_yaml_for(scale);
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

  // The upstream parse_model rounds args[0] (the output channel count) up
  // to the nearest multiple of 8. Internal args (c3/c4 of RepNCSPELAN4
  // etc.) are passed through unchanged. v9m hits this at layers 16/18
  // (180 → 184); v9c/v9t/v9s happen to have all multiples of 8 already.
  auto md8 = [](int v) { return ((v + 7) / 8) * 8; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    int in_ch = in_ch_for(i);
    if (s.kind == "Conv") {
      int c_out = md8(s.args[0]);
      model->push_back(Conv(in_ch, c_out, s.args[1], s.args[2]));
      ch.push_back(c_out);
    } else if (s.kind == "RepNCSPELAN4") {
      int c2 = md8(s.args[0]);
      model->push_back(RepNCSPELAN4(in_ch, c2, s.args[1], s.args[2], s.args[3]));
      ch.push_back(c2);
    } else if (s.kind == "ADown") {
      int c2 = md8(s.args[0]);
      model->push_back(ADown(in_ch, c2));
      ch.push_back(c2);
    } else if (s.kind == "AConv") {
      int c2 = md8(s.args[0]);
      model->push_back(AConv(in_ch, c2));
      ch.push_back(c2);
    } else if (s.kind == "ELAN1") {
      int c2 = md8(s.args[0]);
      model->push_back(ELAN1(in_ch, c2, s.args[1], s.args[2]));
      ch.push_back(c2);
    } else if (s.kind == "SPPELAN") {
      int c2 = md8(s.args[0]);
      model->push_back(SPPELAN(in_ch, c2, s.args[1]));
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
    } else if (s.kind == "Identity") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    } else if (s.kind == "CBLinear") {
      // args = c2s list (one int per branch).
      std::vector<int> c2s_list = s.args;
      model->push_back(CBLinear(in_ch, c2s_list));
      // Sum-of-c2s is the "stored" channel count; downstream CBFuse
      // slices to pick a branch.
      int sum_c = 0;
      for (int c : c2s_list) sum_c += c;
      ch.push_back(sum_c);
    } else if (s.kind == "CBFuse") {
      // args = idx list (one int per non-last input).
      // For each non-last input, the input is a CBLinear's full output;
      // we need its c2s list (stored on that CBLinearImpl).
      std::vector<std::vector<int>> c2s_per_input;
      for (size_t k = 0; k + 1 < s.from.size(); ++k) {
        int idx_layer = resolve_idx(s.from[k], (int)i);
        auto* cb = model[idx_layer]->as<CBLinearImpl>();
        TORCH_CHECK(cb != nullptr,
                    "CBFuse: input ", idx_layer, " is not a CBLinear");
        c2s_per_input.push_back(cb->c2s);
      }
      model->push_back(CBFuse(s.args, c2s_per_input));
      // Output ch = anchor (last input) ch.
      int last_idx = resolve_idx(s.from.back(), (int)i);
      ch.push_back(last_idx == -1 ? c_in_img : ch[last_idx]);
    } else if (s.kind == "Detect") {
      std::vector<int> det_ch;
      for (int f : s.from) det_ch.push_back(ch[f]);
      model->push_back(Detect(nc, det_ch, /*legacy=*/true));
      ch.push_back(0);
    } else {
      throw std::runtime_error("yolo9: unknown layer kind '" + s.kind + "'");
    }
  }
}

torch::Tensor Yolo9Impl::forward_eval(torch::Tensor x) {
  const auto& yaml = v9_yaml_for(scale);
  std::vector<torch::Tensor> outs(yaml.size());
  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    std::vector<torch::Tensor> in_list;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        parts.push_back(idx == -1 ? x : outs[idx]);
      }
      in = torch::cat(parts, /*dim=*/1);
    } else if (s.kind == "CBFuse") {
      // Multiple inputs — collect them as a list (no cat).
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        in_list.push_back(idx == -1 ? x : outs[idx]);
      }
    } else if (s.kind != "Detect") {
      int f = s.from[0];
      int idx = resolve_idx(f, (int)i);
      in = (idx == -1) ? x : outs[idx];
    }

    if (s.kind == "Conv")              outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "RepNCSPELAN4") outs[i] = model[i]->as<RepNCSPELAN4Impl>()->forward(in);
    else if (s.kind == "ADown")        outs[i] = model[i]->as<ADownImpl>()->forward(in);
    else if (s.kind == "AConv")        outs[i] = model[i]->as<AConvImpl>()->forward(in);
    else if (s.kind == "ELAN1")        outs[i] = model[i]->as<ELAN1Impl>()->forward(in);
    else if (s.kind == "SPPELAN")      outs[i] = model[i]->as<SPPELANImpl>()->forward(in);
    else if (s.kind == "Upsample")     outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")       outs[i] = in;
    else if (s.kind == "Identity")     outs[i] = in;
    else if (s.kind == "CBLinear")     outs[i] = model[i]->as<CBLinearImpl>()->forward(in);
    else if (s.kind == "CBFuse")       outs[i] = model[i]->as<CBFuseImpl>()->forward(in_list);
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

std::vector<torch::Tensor> Yolo9Impl::forward_train(torch::Tensor x) {
  // Mirror of forward_eval up to (but not through) the Detect decode — we
  // return d->forward_features(det_in) directly so V8DetectionLoss can
  // consume the per-scale raw [B, 4*reg_max+nc, H_i, W_i] feature maps.
  const auto& yaml = v9_yaml_for(scale);
  std::vector<torch::Tensor> outs(yaml.size());
  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    std::vector<torch::Tensor> in_list;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        parts.push_back(idx == -1 ? x : outs[idx]);
      }
      in = torch::cat(parts, /*dim=*/1);
    } else if (s.kind == "CBFuse") {
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        in_list.push_back(idx == -1 ? x : outs[idx]);
      }
    } else if (s.kind != "Detect") {
      int f = s.from[0];
      int idx = resolve_idx(f, (int)i);
      in = (idx == -1) ? x : outs[idx];
    }

    if (s.kind == "Conv")              outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "RepNCSPELAN4") outs[i] = model[i]->as<RepNCSPELAN4Impl>()->forward(in);
    else if (s.kind == "ADown")        outs[i] = model[i]->as<ADownImpl>()->forward(in);
    else if (s.kind == "AConv")        outs[i] = model[i]->as<AConvImpl>()->forward(in);
    else if (s.kind == "ELAN1")        outs[i] = model[i]->as<ELAN1Impl>()->forward(in);
    else if (s.kind == "SPPELAN")      outs[i] = model[i]->as<SPPELANImpl>()->forward(in);
    else if (s.kind == "Upsample")     outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")       outs[i] = in;
    else if (s.kind == "Identity")     outs[i] = in;
    else if (s.kind == "CBLinear")     outs[i] = model[i]->as<CBLinearImpl>()->forward(in);
    else if (s.kind == "CBFuse")       outs[i] = model[i]->as<CBFuseImpl>()->forward(in_list);
    else if (s.kind == "Detect") {
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
  TORCH_CHECK(false, "Yolo9Impl::forward_train: no Detect layer in yaml");
}

int Yolo9Impl::load_from_state_dict(
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
