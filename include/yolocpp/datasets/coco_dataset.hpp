#pragma once
//
// COCO JSON dataset loader (#54B).
//
// Reads `instances_<split>.json` (the standard COCO 2017 schema):
//
//   {
//     "images":      [{"id": 1, "file_name": "000000.jpg",
//                      "width": W, "height": H}, ...],
//     "annotations": [{"image_id": 1, "category_id": 18,
//                      "bbox": [x, y, w, h], ...}, ...],
//     "categories":  [{"id": 1, "name": "person"}, ...],
//   }
//
// `bbox` is [x_top_left, y_top_left, width, height] in absolute
// pixel coords (NOT YOLO-normalised). We convert to (cx, cy, w, h)
// in [0, 1] at parse time so getitem returns the standard
// `YoloExample` shape.
//
// Class IDs in COCO are sparse (1..90 with gaps). We compress them
// to a 0..N-1 range using the order of `categories[]` in the JSON,
// which is stable across COCO releases. Class names are derived
// from the JSON's `name` field (no need to pass them externally).
//
// Image paths are relative to a user-supplied images directory,
// because the JSON only stores `file_name` (e.g. "000000123456.jpg")
// and downstream consumers all keep images in a separate folder
// alongside the JSON.

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"  // YoloExample, AugConfig, Batch

namespace yolocpp::datasets {

class CocoDataset {
 public:
  // json_path:  path to instances_<split>.json
  // images_dir: directory holding the actual jpgs (e.g.
  //             coco/images/val2017/). Optional — defaults to the
  //             json's parent directory if empty.
  // imgsz:      letterbox target.
  // aug:        augmentation config.
  CocoDataset(std::string json_path, std::string images_dir, int imgsz,
              AugConfig aug = {});

  std::size_t size() const { return img_paths_.size(); }

  YoloExample get(std::size_t idx, std::uint64_t aug_seed = 0) const;

  using Batch = YoloDataset::Batch;
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

  int num_classes() const { return (int)names_.size(); }
  int imgsz()       const { return imgsz_; }
  const std::vector<std::string>& names() const { return names_; }

  // Accessors for the `--data` dispatcher (see `cli::make_dataset`).
  const std::vector<std::string>&   paths()  const { return img_paths_; }
  const std::vector<torch::Tensor>& labels() const { return labels_; }

 private:
  std::vector<std::string>    img_paths_;
  std::vector<torch::Tensor>  labels_;       // [N,5] per image (normalised)
  int                         imgsz_ = 640;
  std::vector<std::string>    names_;
  AugConfig                   aug_;
};

}  // namespace yolocpp::datasets
