#pragma once
//
// yolo1 — Redmon et al., "You Only Look Once: Unified, Real-Time Object
// Detection", CVPR 2016. Original pjreddie Darknet model:
//   • 24 convolutional layers (leaky 0.1, NO batch norm — BN didn't land
//     in YOLO until v2)
//   • 2 fully-connected layers on top (4096 → S·S·(B·5+C) flat output)
//   • Default S=7 grid, B=2 boxes per cell, C=20 classes on PASCAL VOC,
//     input 448×448. Output therefore is `[B, 7*7*(2*5+20)] = [B, 1470]`.
//
// Output layout follows Darknet's `[detection]` storage order — NOT
// interleaved per cell, but three flat blocks back to back:
//   [0           .. S·S·C       )  class probabilities (per cell)
//   [S·S·C       .. S·S·(C+B)   )  per-box objectness scores
//   [S·S·(C+B)   .. S·S·(C+B+4B))  per-box (x, y, w, h)
//
// `forward_eval` does the Darknet → "yolo standard" reshape + decode
// internally and returns `[B, 4+C, A]` (xyxy in input pixels, sigmoided-
// class scores, where A = S·S·B = 98), drop-in for `inference::nms`.
//

#include <torch/torch.h>

namespace yolocpp::models {

// One Darknet [convolutional] block with batch_normalize=0: a single
// Conv2d (bias=true) followed by leaky(0.1). Module names match what
// the .weights walker expects (a bare Conv2d child under each block).
struct ConvLeakyNoBNImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  ConvLeakyNoBNImpl(int c_in, int c_out, int k, int s);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvLeakyNoBN);

// Original "YOLO" — full 24-conv model (Redmon Table 1).
struct Yolo1Impl : torch::nn::Module {
  // Backbone: 24 conv layers (no BN) emitting a (1024, 7, 7) feature map
  // from a (3, 448, 448) input.
  torch::nn::Sequential backbone{nullptr};
  // Head: two fully-connected layers (4096-hidden → S·S·(B·5+C) output).
  torch::nn::Linear     fc1{nullptr};
  torch::nn::Linear     fc2{nullptr};
  // Optional dropout between the two FCs (Darknet's `[dropout]` block).
  torch::nn::Dropout    drop{nullptr};

  int S  = 7;   // grid side
  int B  = 2;   // boxes per cell
  int nc = 20;  // class count (VOC by default)
  int imgsz_default = 448;
  // Effective stride (input_size / grid_side). Populated lazily on
  // the first `forward_train` call. Mirrors the convention every
  // other model uses so the trainer's templated body can read it
  // without a holder-specific branch.
  std::vector<double> stride;

  explicit Yolo1Impl(int nc = 20, int S = 7, int B = 2);

  // Returns the raw flat tensor `[B, S·S·(B·5+nc)]` — exactly what
  // Darknet's connected layer emits. Used by the weight-converter
  // round-trip test.
  torch::Tensor forward_flat(torch::Tensor x);

  // Returns `[B, 4+nc, A]` in xyxy + sigmoid-class form, A = S·S·B.
  // Drop-in for `inference::nms`.
  torch::Tensor forward_eval(torch::Tensor x);

  // Returns `{flat}` where flat is `[B, S·S·(B·5+nc)]` — the raw
  // post-FC output. Single-element vector so the templated trainer
  // can iterate over "pyramid levels" generically.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  // Match-by-name copy from a flat state_dict (the converted .pt
  // emitted by `convert_yolov1_weights`). Returns the number of
  // tensors successfully copied.
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo1);

}  // namespace yolocpp::models
