#pragma once
//
// Pascal VOC dataset loader (#54B).
//
// Standard VOC layout (PASCAL VOC 2007 / 2012, also reused by many
// downstream forks):
//
//   <root>/JPEGImages/<id>.jpg
//   <root>/Annotations/<id>.xml          — per-image XML with <object>
//                                          entries (bounding box in
//                                          absolute pixel coords)
//   <root>/ImageSets/Main/<split>.txt    — one image-id per line
//
// Each XML's `<object>` block:
//   <object>
//     <name>person</name>          — class name (looked up against `names`)
//     <bndbox>
//       <xmin>23</xmin>            — pixel coords, 1-indexed in VOC
//       <ymin>200</ymin>
//       <xmax>456</xmax>
//       <ymax>720</ymax>
//     </bndbox>
//   </object>
//
// We convert every box to YOLO-normalised (cls, cx, cy, w, h) at
// parse time so getitem() returns the same `YoloExample` shape the
// rest of the pipeline expects. Class names default to the standard
// VOC 20 (aeroplane, bicycle, bird, ..., tvmonitor); pass a custom
// `names` to override.

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <random>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"  // YoloExample, AugConfig, Batch

namespace yolocpp::datasets {

// The 20 canonical VOC class names in their conventional order.
const std::vector<std::string>& voc_default_names();

class VocDataset {
 public:
  // root:   VOC root directory (must contain JPEGImages/, Annotations/,
  //         ImageSets/Main/).
  // split:  e.g. "train", "val", "trainval", "test" — must match a
  //         file in ImageSets/Main/.
  // imgsz:  letterbox target.
  // names:  class names in index order. Default: voc_default_names().
  // aug:    augmentation config.
  VocDataset(std::string root, std::string split, int imgsz,
             std::vector<std::string> names = voc_default_names(),
             AugConfig aug = {});

  std::size_t size() const { return img_paths_.size(); }

  YoloExample get(std::size_t idx, std::uint64_t aug_seed = 0) const;

  using Batch = YoloDataset::Batch;
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

  int num_classes() const { return (int)names_.size(); }
  int imgsz()       const { return imgsz_; }
  const std::vector<std::string>& names() const { return names_; }

 private:
  std::vector<std::string>    img_paths_;
  std::vector<torch::Tensor>  labels_;          // [N,5] per image (normalised)
  int                         imgsz_ = 640;
  std::vector<std::string>    names_;
  AugConfig                   aug_;
};

}  // namespace yolocpp::datasets
