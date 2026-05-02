#pragma once
//
// Flat single-file dataset format (#54A).
//
// One CSV/TSV file holds every image + every label across every
// split. Schema (header line required, fields whitespace- /
// comma- / tab-separated — auto-detected from the first row):
//
//   split  image_path  class_id  x_center  y_center  width  height
//
// where:
//   - `split`      ∈ {train, val, test}     — which fold this row belongs to.
//   - `image_path` is absolute, or relative to the dataset file's directory.
//   - `class_id` is an integer class index. Empty cell ⇒ background image
//     (no objects); the row exists to register the image in the split.
//   - `x_center, y_center, width, height` are normalised to [0, 1] in the
//     image's coordinate frame — same convention as the upstream YOLO
//     `<root>/labels/<split>/<id>.txt` per-line format.
//
// Multiple labels for one image map to multiple rows that share the
// same `image_path`. The dataset groups rows by image at construction
// time so downstream getitem returns the standard
// `(img [3,H,W], targets [N,5])` example.
//
// **Why this format?** Single file is easier to commit to revision
// control, easier to splice + shuffle by `--seed`, and trivial to
// share between people / experiments. The split column makes
// train/val/test boundaries part of the data — no more "did we
// shuffle the same way" debugging. Pascal VOC and COCO JSON loaders
// (#54B) ultimately fold into this same in-memory representation.
//
// Class names: passed to the constructor (typically from `--names`
// or a sibling YAML). The flat file itself stores integer class IDs.

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"  // YoloExample, AugConfig, Batch shape

namespace yolocpp::datasets {

class FlatDataset {
 public:
  // file_path: path to the .csv / .tsv. Header line required.
  // split:     one of {"train", "val", "test"}; rows whose `split`
  //            column doesn't match are filtered out.
  // imgsz:     letterbox target.
  // names:     class names (used for nc).
  // aug:       augmentation config — same struct YoloDataset uses.
  // seed:      0 ⇒ stable per-idx augmentation (matches YoloDataset);
  //            non-zero seeds the per-call augmentation RNG.
  FlatDataset(std::string file_path, std::string split, int imgsz,
              std::vector<std::string> names, AugConfig aug = {},
              std::uint64_t seed = 0);

  std::size_t size() const { return img_paths_.size(); }

  // Same return shape as YoloDataset::get for downstream compatibility.
  YoloExample get(std::size_t idx, std::uint64_t aug_seed = 0) const;

  // Same Batch shape as YoloDataset so trainer + validator can
  // accept either dataset interchangeably (#54B will unify under a
  // common base; for #54A we keep the type concrete).
  using Batch = YoloDataset::Batch;
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

  int        num_classes() const { return (int)names_.size(); }
  int        imgsz()       const { return imgsz_; }
  const std::vector<std::string>& names() const { return names_; }

  // Accessors for the `--data` dispatcher (see `cli::make_dataset`).
  const std::vector<std::string>&   paths()  const { return img_paths_; }
  const std::vector<torch::Tensor>& labels() const { return labels_; }

 private:
  // After parsing, one entry per image; labels_[i] is an [N,5]
  // tensor with normalised (cls, cx, cy, w, h) — shape matches what
  // YoloDataset's load_targets returns.
  std::vector<std::string>    img_paths_;
  std::vector<torch::Tensor>  labels_;
  int                         imgsz_ = 640;
  std::vector<std::string>    names_;
  AugConfig                   aug_;
  std::uint64_t               seed_ = 0;
};

}  // namespace yolocpp::datasets
