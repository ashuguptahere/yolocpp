#pragma once
//
// YOLO-format dataset loader.
//
// Filesystem layout (matches the upstream YOLO dataset convention):
//   <root>/images/<split>/<id>.jpg
//   <root>/labels/<split>/<id>.txt
// where each .txt line is "<cls> <cx> <cy> <w> <h>" with all coords
// normalized to [0, 1].
//
// The dataset returns a per-image example:
//   img: float32 [3, H, W] in [0, 1] — already letterboxed to imgsz × imgsz
//   targets: float32 [N, 5] (cls, cx, cy, w, h) in absolute pixel coords
//            of the letterboxed image.
//
// Augmentation (deterministic when augment=false):
//   • letterbox (always)
//   • horizontal flip with prob `flip_p`
//   • HSV jitter (h, s, v) with given amplitudes
// Mosaic / mixup are TODO.

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <random>
#include <string>
#include <utility>
#include <vector>

namespace yolocpp::datasets {

struct AugConfig {
  bool   augment = true;
  float  flip_p  = 0.5f;
  float  hsv_h   = 0.015f;
  float  hsv_s   = 0.7f;
  float  hsv_v   = 0.4f;
  // If true, letterbox pads only enough to make each side a multiple of
  // 32 (matches the upstream val `rect=True`). Image sizes vary per sample;
  // safe for batch_size=1 validation. Off by default for training.
  bool   rect    = false;
  // Mosaic-4 + Mixup probabilities (#54D). Both default off so
  // existing call sites are unaffected. Mosaic stitches 4 sampled
  // images into a 2x2 grid + crops to imgsz×imgsz, translating
  // bboxes; the upstream YOLO recipe enables it for the bulk of
  // training and disables in the last ~10 epochs ("close_mosaic").
  // Mixup blends two examples with alpha ~ Beta(8, 8) and
  // concatenates their bbox lists. mosaic_p is checked first; mixup
  // is applied to the resulting sample.
  float  mosaic_p = 0.0f;
  float  mixup_p  = 0.0f;
};

struct YoloExample {
  torch::Tensor img;       // [3, H, W] float32
  torch::Tensor targets;   // [N, 5]   (cls, cx, cy, w, h) — pixel coords
  std::string   img_path;
  int           orig_w = 0, orig_h = 0;
  // For validation: gain + pad applied during letterbox so detections can
  // be unscaled back to original-image coords.
  double        gain = 1.0, pad_x = 0.0, pad_y = 0.0;
};

class YoloDataset {
 public:
  // root:   dataset root directory.
  // split:  e.g. "train", "val".
  // imgsz:  letterbox target.
  // names:  class names (used for nc).
  // aug:    augmentation config.
  YoloDataset(std::string root, std::string split, int imgsz,
              std::vector<std::string> names, AugConfig aug = {});

  // From explicit image+label path lists (useful for tiny test datasets).
  YoloDataset(std::vector<std::string> img_paths,
              std::vector<std::string> lbl_paths,
              int imgsz, std::vector<std::string> names,
              AugConfig aug = {});

  std::size_t size() const { return img_paths_.size(); }

  // Get a single example. Thread-safe (uses a per-call RNG seed).
  YoloExample get(std::size_t idx, uint64_t aug_seed = 0) const;

  // Build a batch of `bsz` examples by uniformly sampling indices.
  // Returns a tuple (imgs [B, 3, H, W], targets [M, 6] with batch index).
  // Targets format: (batch_idx, cls, cx, cy, w, h) — pixel coords.
  struct Batch {
    torch::Tensor              imgs;
    torch::Tensor              targets;
    std::vector<YoloExample>   examples;
  };
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

  int  num_classes() const { return (int)names_.size(); }
  int  imgsz()       const { return imgsz_; }
  const std::vector<std::string>& names() const { return names_; }

 private:
  std::vector<std::string>  img_paths_;
  std::vector<std::string>  lbl_paths_;
  int                       imgsz_ = 640;
  std::vector<std::string>  names_;
  AugConfig                 aug_;
};

}  // namespace yolocpp::datasets
