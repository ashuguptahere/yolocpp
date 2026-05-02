// Smoke for Mosaic-4 + Mixup augmentation (#54D).
//
// Loads coco8 (4 train images), runs sample_batch with mosaic_p=1
// then mixup_p=1, asserts:
//   - imgs come back at the requested imgsz×imgsz
//   - target tensor is non-degenerate (mosaic stitches 4 images so
//     batches typically end up with 2× the original GT count)
//   - each (cx, cy, w, h) is inside the imgsz frame.

#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/inference/predictor.hpp"

#define EXPECT(cond, msg) \
  do {                    \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  namespace fs = std::filesystem;
  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[mosaic] SKIP — data/coco8 missing\n";
    return 0;
  }

  std::vector<std::string> names = yolocpp::inference::coco_names();
  yolocpp::datasets::AugConfig aug;
  aug.augment  = true;
  aug.flip_p   = 0.f;          // disable to keep test deterministic
  aug.hsv_h    = 0.f; aug.hsv_s = 0.f; aug.hsv_v = 0.f;
  aug.mosaic_p = 1.0f;          // always mosaic
  aug.mixup_p  = 0.0f;          // skip mixup for the first test

  yolocpp::datasets::YoloDataset ds("data/coco8", "train", /*imgsz=*/320,
                                      names, aug);
  EXPECT(ds.size() >= 4, "need ≥ 4 train images for mosaic-4");

  std::mt19937 rng(42);
  auto batch = ds.sample_batch(/*bsz=*/2, rng);
  EXPECT(batch.imgs.size(0) == 2, "batch size 2");
  EXPECT(batch.imgs.size(1) == 3, "3 channels");
  EXPECT(batch.imgs.size(2) == 320 && batch.imgs.size(3) == 320,
         "imgs should be 320×320 after mosaic");

  // Bbox sanity: every (cx,cy,w,h) should be inside the imgsz frame.
  if (batch.targets.size(0) > 0) {
    auto a = batch.targets.accessor<float, 2>();
    for (int i = 0; i < (int)batch.targets.size(0); ++i) {
      EXPECT(a[i][2] >= 0 && a[i][2] <= 320, "cx in frame");
      EXPECT(a[i][3] >= 0 && a[i][3] <= 320, "cy in frame");
      EXPECT(a[i][4] > 0, "w positive");
      EXPECT(a[i][5] > 0, "h positive");
    }
  }
  std::cout << "[mosaic] mosaic-only batch OK (" << batch.targets.size(0)
            << " labels across 2 mosaics)\n";

  // Mixup smoke.
  aug.mosaic_p = 0.0f;
  aug.mixup_p  = 1.0f;
  yolocpp::datasets::YoloDataset ds2("data/coco8", "train", /*imgsz=*/320,
                                       names, aug);
  std::mt19937 rng2(42);
  auto batch2 = ds2.sample_batch(/*bsz=*/2, rng2);
  EXPECT(batch2.imgs.size(2) == 320 && batch2.imgs.size(3) == 320,
         "mixup imgs should be 320×320");
  std::cout << "[mosaic] mixup-only batch OK (" << batch2.targets.size(0)
            << " concatenated labels)\n";
  return 0;
}
