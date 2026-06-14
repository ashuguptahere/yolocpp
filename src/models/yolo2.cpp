#include "yolocpp/models/yolo2.hpp"

#include "yolocpp/models/yolo4.hpp"   // ConvLeaky / ConvLeakyImpl (Conv+BN+leaky)

namespace yolocpp::models {

namespace {

torch::nn::MaxPool2d mp2() {
  return torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(2).stride(2));
}

// Darknet's last maxpool in yolov2-tiny is stride-1 with kernel-2 plus a
// 1-pixel right/bottom pad. We emulate it with explicit constant_pad +
// max_pool2d(k=2, s=1).
torch::Tensor mp_stride1_pad(const torch::Tensor& x) {
  auto padded = torch::constant_pad_nd(x, {0, 1, 0, 1}, /*value=*/0);
  return torch::max_pool2d(padded, /*kernel=*/{2, 2}, /*stride=*/{1, 1});
}

}  // namespace

// Darknet's `[reorg]` (stride=2, no reverse) — turns (B, C, H, W) into
// (B, C·s², H/s, W/s) with the exact flat-memory element order Darknet
// produces, so the subsequent conv weights (which were trained against
// that layout) consume the right channels.
//
// Concretely: input channel `offset*out_c + c2` (with out_c=C/s²,
// offset ∈ [0, s²), c2 ∈ [0, out_c)) lands at output channel `c2` in
// the (out_c, H·s, W·s) intermediate buffer at strided position
// (dy = offset/s, dx = offset%s). Re-interpreting that buffer as
// (C·s², H/s, W/s) is then a plain `view` — no copy.
torch::Tensor reorg(const torch::Tensor& x, int stride) {
  const int64_t B = x.size(0);
  const int64_t C = x.size(1);
  const int64_t H = x.size(2);
  const int64_t W = x.size(3);
  TORCH_CHECK(H % stride == 0 && W % stride == 0,
              "reorg: H, W must be divisible by stride");
  TORCH_CHECK(C % (stride * stride) == 0,
              "reorg: C must be divisible by stride²");
  const int64_t out_c = C / (stride * stride);
  auto out = torch::empty({B, out_c, H * stride, W * stride}, x.options());
  for (int offset = 0; offset < stride * stride; ++offset) {
    const int dy = offset / stride;
    const int dx = offset % stride;
    auto src = x.slice(1, offset * out_c, (offset + 1) * out_c);  // [B, out_c, H, W]
    out.slice(2, dy, H * stride, stride)
       .slice(3, dx, W * stride, stride)
       .copy_(src);
  }
  return out.view({B, C * stride * stride, H / stride, W / stride});
}

std::vector<float> yolo2_voc_anchors() {
  return {1.3221f, 1.73145f,
          3.19275f, 4.00944f,
          5.05587f, 8.09892f,
          9.47112f, 4.84053f,
          11.2364f, 10.0071f};
}

std::vector<float> yolo2_coco_anchors() {
  return {0.57273f, 0.677385f,
          1.87446f, 2.06253f,
          3.33843f, 5.47434f,
          7.88282f, 3.52778f,
          9.77052f, 9.16828f};
}

// yolov2-tiny-voc.cfg ships its OWN k-means anchor set — distinct from full
// yolov2-voc. (yolov2-tiny.cfg for COCO, by contrast, reuses the full COCO
// anchors, so there is no separate tiny-coco set.)
std::vector<float> yolo2_tiny_voc_anchors() {
  return {1.08f,  1.19f,
          3.42f,  4.41f,
          6.63f,  11.38f,
          9.42f,  5.11f,
          16.62f, 10.52f};
}

Yolo2Impl::Yolo2Impl(Yolo2Scale s, int nc_, std::vector<float> anchors_)
    : scale(s), nc(nc_), anchors(std::move(anchors_)) {
  if (anchors.empty()) {
    if (nc == 80)
      anchors = yolo2_coco_anchors();          // tiny + full COCO share anchors
    else if (scale == Yolo2Scale::Tiny)
      anchors = yolo2_tiny_voc_anchors();       // tiny-voc has its own set
    else
      anchors = yolo2_voc_anchors();
  }
  const int na  = num_anchors();
  const int out_c = na * (5 + nc);

  if (scale == Yolo2Scale::Tiny) {
    // yolov2-tiny.cfg: 9 ConvLeaky + 6 maxpool + 1×1 head. Split into
    // three Sequential groups so we can inject the fake-stride pool
    // between groups in `forward_raw`.
    //   t_a: ConvLeaky 16, mp, ConvLeaky 32, mp, ConvLeaky 64, mp,
    //         ConvLeaky 128, mp, ConvLeaky 256, mp, ConvLeaky 512
    //   (then mp_stride1_pad — between groups)
    //   t_b: ConvLeaky 1024, ConvLeaky 1024, Conv2d 1×1 (no BN)
    torch::nn::Sequential t;
    t->push_back(ConvLeaky(  3,   16, 3, 1));   t->push_back(mp2());
    t->push_back(ConvLeaky( 16,   32, 3, 1));   t->push_back(mp2());
    t->push_back(ConvLeaky( 32,   64, 3, 1));   t->push_back(mp2());
    t->push_back(ConvLeaky( 64,  128, 3, 1));   t->push_back(mp2());
    t->push_back(ConvLeaky(128,  256, 3, 1));   t->push_back(mp2());
    t->push_back(ConvLeaky(256,  512, 3, 1));
    t->push_back(ConvLeaky(512, 1024, 3, 1));    // injected mp_stride1 happens before this in forward
    t->push_back(ConvLeaky(1024,1024, 3, 1));
    t->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1024, out_c, 1).bias(true)));
    tiny = register_module("tiny", t);
    return;
  }

  // Full Darknet-19 backbone.
  torch::nn::Sequential e;
  e->push_back(ConvLeaky(  3,   32, 3, 1));  e->push_back(mp2());
  e->push_back(ConvLeaky( 32,   64, 3, 1));  e->push_back(mp2());
  e->push_back(ConvLeaky( 64,  128, 3, 1));
  e->push_back(ConvLeaky(128,   64, 1, 1));
  e->push_back(ConvLeaky( 64,  128, 3, 1));  e->push_back(mp2());
  e->push_back(ConvLeaky(128,  256, 3, 1));
  e->push_back(ConvLeaky(256,  128, 1, 1));
  e->push_back(ConvLeaky(128,  256, 3, 1));  e->push_back(mp2());
  e->push_back(ConvLeaky(256,  512, 3, 1));
  e->push_back(ConvLeaky(512,  256, 1, 1));
  e->push_back(ConvLeaky(256,  512, 3, 1));
  e->push_back(ConvLeaky(512,  256, 1, 1));
  e->push_back(ConvLeaky(256,  512, 3, 1));   // conv #17 — passthrough source (26×26×512)
  early = register_module("early", e);

  torch::nn::Sequential l;
  l->push_back(mp2());
  l->push_back(ConvLeaky( 512, 1024, 3, 1));
  l->push_back(ConvLeaky(1024,  512, 1, 1));
  l->push_back(ConvLeaky( 512, 1024, 3, 1));
  l->push_back(ConvLeaky(1024,  512, 1, 1));
  l->push_back(ConvLeaky( 512, 1024, 3, 1));  // 13×13×1024 — last backbone conv
  late = register_module("late", l);

  torch::nn::Sequential hp;
  hp->push_back(ConvLeaky(1024, 1024, 3, 1));
  hp->push_back(ConvLeaky(1024, 1024, 3, 1));
  head_pre = register_module("head_pre", hp);

  torch::nn::Sequential hpt;
  hpt->push_back(ConvLeaky(512, 64, 1, 1));   // passthrough bottleneck
  head_pt = register_module("head_pt", hpt);

  torch::nn::Sequential hpost;
  hpost->push_back(ConvLeaky(256 + 1024, 1024, 3, 1));
  hpost->push_back(torch::nn::Conv2d(
      torch::nn::Conv2dOptions(1024, out_c, 1).bias(true)));  // final 1×1, no BN
  head_post = register_module("head_post", hpost);
}

torch::Tensor Yolo2Impl::forward_raw(torch::Tensor x) {
  if (scale == Yolo2Scale::Tiny) {
    // Walk `tiny` manually so we can inject the stride-1 padded maxpool
    // before the 7th ConvLeaky (index 11 in our push_back order).
    auto y = x;
    auto items = tiny->children();
    for (size_t i = 0; i < items.size(); ++i) {
      if (i == 11) y = mp_stride1_pad(y);
      auto child = items[i];
      if (auto cl = std::dynamic_pointer_cast<ConvLeakyImpl>(child)) {
        y = cl->forward(y);
      } else if (auto mp = std::dynamic_pointer_cast<torch::nn::MaxPool2dImpl>(child)) {
        y = mp->forward(y);
      } else if (auto cv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(child)) {
        y = cv->forward(y);
      } else {
        TORCH_CHECK(false, "yolo2 tiny: unexpected child type at index ", i);
      }
    }
    return y;
  }
  auto e26 = early->forward(x);                  // [B, 512, 26, 26]
  auto p13 = late->forward(e26);                 // [B, 1024, 13, 13]
  auto pre = head_pre->forward(p13);             // [B, 1024, 13, 13]
  auto pt  = head_pt->forward(e26);              // [B, 64, 26, 26]
  auto pt13 = reorg(pt, /*stride=*/2);           // [B, 256, 13, 13]
  auto cat = torch::cat({pt13, pre}, /*dim=*/1); // [B, 1280, 13, 13]
  return head_post->forward(cat);                // [B, na*(5+nc), 13, 13]
}

std::vector<torch::Tensor> Yolo2Impl::forward_train(torch::Tensor x) {
  auto raw = forward_raw(x);
  if (stride.empty()) {
    stride = {(double)x.size(2) / (double)raw.size(2)};
  }
  return {raw};
}

torch::Tensor Yolo2Impl::forward_eval(torch::Tensor x) {
  const int imgH = (int)x.size(2);
  const int imgW = (int)x.size(3);
  auto raw = forward_raw(x);
  const int64_t Bsz     = raw.size(0);
  const int64_t H       = raw.size(2);
  const int64_t W       = raw.size(3);
  const int     na      = num_anchors();
  const int     entries = 5 + nc;

  // [B, na, 5+nc, H, W] → [B, na, H, W, 5+nc].
  auto y = raw.view({Bsz, na, entries, H, W}).permute({0, 1, 3, 4, 2}).contiguous();

  auto cy = torch::arange(H, raw.options()).view({1, 1, H, 1, 1});
  auto cx = torch::arange(W, raw.options()).view({1, 1, 1, W, 1});

  auto anchor_t = torch::from_blob(
      anchors.data(), {na, 2},
      torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .to(raw.device());                  // [na, 2]

  auto tx = y.slice(-1, 0, 1);
  auto ty = y.slice(-1, 1, 2);
  auto tw = y.slice(-1, 2, 3);
  auto th = y.slice(-1, 3, 4);
  auto to = y.slice(-1, 4, 5);
  auto tc = y.slice(-1, 5, entries);

  auto pw = anchor_t.select(-1, 0).view({1, na, 1, 1, 1});
  auto ph = anchor_t.select(-1, 1).view({1, na, 1, 1, 1});

  auto bx = (torch::sigmoid(tx) + cx) / (float)W * (float)imgW;
  auto by = (torch::sigmoid(ty) + cy) / (float)H * (float)imgH;
  auto bw = pw * torch::exp(tw) / (float)W * (float)imgW;
  auto bh = ph * torch::exp(th) / (float)H * (float)imgH;
  auto obj    = torch::sigmoid(to);                              // [B, na, H, W, 1]
  auto cls    = torch::softmax(tc, /*dim=*/-1);                  // [B, na, H, W, nc]
  auto scores = (obj * cls).clamp(0, 1);                         // [B, na, H, W, nc]

  auto x1 = (bx - bw * 0.5f).clamp(0, (float)imgW);
  auto y1 = (by - bh * 0.5f).clamp(0, (float)imgH);
  auto x2 = (bx + bw * 0.5f).clamp(0, (float)imgW);
  auto y2 = (by + bh * 0.5f).clamp(0, (float)imgH);
  auto coords = torch::cat({x1, y1, x2, y2}, /*dim=*/-1);        // [B, na, H, W, 4]

  const int64_t A = (int64_t)na * H * W;
  coords = coords.reshape({Bsz, A, 4});
  scores = scores.reshape({Bsz, A, nc});
  return torch::cat({coords, scores}, /*dim=*/-1)
            .transpose(1, 2)
            .contiguous();
}

int Yolo2Impl::load_from_state_dict(
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
