#include "yolocpp/models/yolo1.hpp"

namespace yolocpp::models {

ConvLeakyNoBNImpl::ConvLeakyNoBNImpl(int c_in, int c_out, int k, int s) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(k / 2).bias(true)));
}
torch::Tensor ConvLeakyNoBNImpl::forward(torch::Tensor x) {
  return torch::leaky_relu(conv(x), 0.1);
}

namespace {

torch::nn::MaxPool2d mp2() {
  return torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(2).stride(2));
}

}  // namespace

Yolo1Impl::Yolo1Impl(int nc_, int S_, int B_) : S(S_), B(B_), nc(nc_) {
  // Backbone — order matches Darknet's yolov1.cfg conv block sequence
  // exactly. Each conv is a [convolutional] block with
  // batch_normalize=0, activation=leaky.
  torch::nn::Sequential bb;
  //   #1  conv 64  / 7×7 / s=2
  bb->push_back(ConvLeakyNoBN(3, 64, 7, 2));
  //   #2  maxpool /2
  bb->push_back(mp2());
  //   #3  conv 192 / 3×3
  bb->push_back(ConvLeakyNoBN(64, 192, 3, 1));
  //   #4  maxpool /2
  bb->push_back(mp2());
  //   #5  conv 128 / 1×1
  bb->push_back(ConvLeakyNoBN(192, 128, 1, 1));
  //   #6  conv 256 / 3×3
  bb->push_back(ConvLeakyNoBN(128, 256, 3, 1));
  //   #7  conv 256 / 1×1
  bb->push_back(ConvLeakyNoBN(256, 256, 1, 1));
  //   #8  conv 512 / 3×3
  bb->push_back(ConvLeakyNoBN(256, 512, 3, 1));
  //   #9  maxpool /2
  bb->push_back(mp2());
  //   #10-#17  four repeats of (conv 256 1×1, conv 512 3×3)
  for (int i = 0; i < 4; ++i) {
    bb->push_back(ConvLeakyNoBN(512, 256, 1, 1));
    bb->push_back(ConvLeakyNoBN(256, 512, 3, 1));
  }
  //   #18  conv 512 / 1×1
  bb->push_back(ConvLeakyNoBN(512, 512, 1, 1));
  //   #19  conv 1024 / 3×3
  bb->push_back(ConvLeakyNoBN(512, 1024, 3, 1));
  //   #20  maxpool /2
  bb->push_back(mp2());
  //   #21-#24  two repeats of (conv 512 1×1, conv 1024 3×3)
  for (int i = 0; i < 2; ++i) {
    bb->push_back(ConvLeakyNoBN(1024, 512, 1, 1));
    bb->push_back(ConvLeakyNoBN(512, 1024, 3, 1));
  }
  //   #25  conv 1024 / 3×3
  bb->push_back(ConvLeakyNoBN(1024, 1024, 3, 1));
  //   #26  conv 1024 / 3×3 / s=2
  bb->push_back(ConvLeakyNoBN(1024, 1024, 3, 2));
  //   #27  conv 1024 / 3×3
  bb->push_back(ConvLeakyNoBN(1024, 1024, 3, 1));
  //   #28  conv 1024 / 3×3
  bb->push_back(ConvLeakyNoBN(1024, 1024, 3, 1));
  backbone = register_module("backbone", bb);

  const int feat_in = 1024 * S * S;     // 1024 · 7 · 7 = 50176
  const int hidden  = 4096;
  const int outputs = S * S * (B * 5 + nc);  // 7·7·(2·5+20) = 1470

  fc1  = register_module("fc1",  torch::nn::Linear(feat_in, hidden));
  drop = register_module("drop", torch::nn::Dropout(
      torch::nn::DropoutOptions(0.5)));
  fc2  = register_module("fc2",  torch::nn::Linear(hidden, outputs));
}

torch::Tensor Yolo1Impl::forward_flat(torch::Tensor x) {
  auto f = backbone->forward(x);
  f = f.flatten(/*start_dim=*/1);
  f = torch::leaky_relu(fc1(f), 0.1);
  f = drop(f);
  return fc2(f);   // [B, S·S·(B·5+nc)]
}

torch::Tensor Yolo1Impl::forward_eval(torch::Tensor x) {
  const int H = (int)x.size(2);
  const int W = (int)x.size(3);
  auto flat   = forward_flat(x);
  const auto Bsz = flat.size(0);

  // Darknet detection output: three contiguous blocks.
  const int64_t SS = (int64_t)S * S;
  const int64_t cls_sz  = SS * nc;
  const int64_t conf_sz = SS * B;
  const int64_t coord_sz = SS * B * 4;

  auto cls_flat   = flat.slice(1, 0, cls_sz)
                        .view({Bsz, SS, nc});                  // [B, SS, nc]
  auto conf_flat  = flat.slice(1, cls_sz, cls_sz + conf_sz)
                        .view({Bsz, SS, B});                   // [B, SS, B]
  auto coord_flat = flat.slice(1, cls_sz + conf_sz,
                                  cls_sz + conf_sz + coord_sz)
                        .view({Bsz, SS, B, 4});                // [B, SS, B, 4]

  // Grid coordinates (cy, cx) for each SS cell.
  auto cy = torch::arange(S, flat.options()).repeat_interleave(S);  // [SS]
  auto cx = torch::arange(S, flat.options()).repeat({S});           // [SS]

  // Decode: bx = (cx + tx) / S × W, etc. v1 emits w, h directly in
  // [0, 1] of the input image (no sqrt at inference — sqrt is in the
  // loss only).
  auto tx = coord_flat.select(-1, 0);  // [B, SS, B]
  auto ty = coord_flat.select(-1, 1);
  auto tw = coord_flat.select(-1, 2);
  auto th = coord_flat.select(-1, 3);

  auto bx = (tx + cx.view({1, SS, 1})) / (float)S * (float)W;
  auto by = (ty + cy.view({1, SS, 1})) / (float)S * (float)H;
  auto bw = tw.clamp_min(0).pow(2) * (float)W;   // v1 outputs sqrt(w), sqrt(h)
  auto bh = th.clamp_min(0).pow(2) * (float)H;   // → unsqrt + clamp_min(0) here

  auto x1 = (bx - bw * 0.5f).clamp(0, (float)W);
  auto y1 = (by - bh * 0.5f).clamp(0, (float)H);
  auto x2 = (bx + bw * 0.5f).clamp(0, (float)W);
  auto y2 = (by + bh * 0.5f).clamp(0, (float)H);

  // Class scores per (cell, box): conf · class_prob (v1 trains class
  // probs as softmax-like targets; at inference the head emits them
  // post-something — Darknet's reference impl multiplies the raw
  // values without an extra non-linearity. We mirror that.)
  auto class_per_cell = cls_flat.unsqueeze(2);                 // [B, SS, 1, nc]
  auto conf           = conf_flat.unsqueeze(-1);               // [B, SS, B, 1]
  auto scores         = (conf * class_per_cell)
                            .clamp(0, 1);                      // [B, SS, B, nc]

  // Pack to [B, 4+nc, A] with A = SS·B.
  auto coords = torch::stack({x1, y1, x2, y2}, /*dim=*/-1);    // [B, SS, B, 4]
  const int64_t A = SS * B;
  coords = coords.reshape({Bsz, A, 4});                        // [B, A, 4]
  scores = scores.reshape({Bsz, A, nc});                       // [B, A, nc]
  auto out = torch::cat({coords, scores}, /*dim=*/-1)          // [B, A, 4+nc]
                .transpose(1, 2)
                .contiguous();                                 // [B, 4+nc, A]
  return out;
}

int Yolo1Impl::load_from_state_dict(
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
