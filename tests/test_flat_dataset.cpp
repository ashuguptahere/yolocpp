// Smoke for FlatDataset (#54A): synthesise a tiny CSV that points
// at data/bus.jpg, verify split filtering + getitem returns a
// well-shaped (img, targets) pair.

#include <filesystem>
#include <fstream>
#include <iostream>

#include "yolocpp/datasets/flat_dataset.hpp"

#define EXPECT(cond, msg) \
  do {                    \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  namespace fs = std::filesystem;
  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[flat] SKIP — data/bus.jpg missing\n";
    return 0;
  }

  // Write a tiny CSV with three labels across two splits. Use the
  // absolute bus.jpg path so resolution is unambiguous regardless
  // of where the test binary is invoked from.
  auto bus = fs::absolute("data/bus.jpg").string();
  fs::path tmp = "/tmp/yolocpp_flat_test.csv";
  {
    std::ofstream o(tmp);
    o << "split,image_path,class_id,x_center,y_center,width,height\n";
    o << "train," << bus << ",0,0.5,0.5,0.3,0.4\n";
    o << "train," << bus << ",5,0.4,0.6,0.6,0.5\n";
    o << "val,"   << bus << ",2,0.2,0.2,0.1,0.1\n";
    // Background image (empty cls) — registers no label.
    o << "train," << bus << ",,,,,\n";
  }

  std::vector<std::string> names = {"a", "b", "c", "d", "e", "f"};
  yolocpp::datasets::AugConfig aug;
  aug.augment = false;

  // Train split: 1 image with 2 labels.
  yolocpp::datasets::FlatDataset train(tmp.string(), "train", /*imgsz=*/320,
                                         names, aug, /*seed=*/42);
  EXPECT(train.size() == 1, "train split should have 1 image");
  auto ex = train.get(0);
  EXPECT(ex.img.dim() == 3, "img should be [C,H,W]");
  EXPECT(ex.img.size(0) == 3, "img should have 3 channels");
  EXPECT(ex.img.size(1) == 320 && ex.img.size(2) == 320,
         "img should be letterboxed to 320x320");
  EXPECT(ex.targets.size(0) == 2, "train image should have 2 labels");
  EXPECT(ex.targets.size(1) == 5, "targets should be [N,5]");

  auto a = ex.targets.accessor<float, 2>();
  for (int i = 0; i < (int)ex.targets.size(0); ++i) {
    EXPECT(a[i][0] >= 0 && a[i][0] < (float)names.size(),
           "class id out of range");
    EXPECT(a[i][1] > 0 && a[i][1] < 320, "cx out of range");
    EXPECT(a[i][2] > 0 && a[i][2] < 320, "cy out of range");
    EXPECT(a[i][3] > 0 && a[i][4] > 0, "w/h must be positive");
  }

  // Val split: 1 image with 1 label.
  yolocpp::datasets::FlatDataset val(tmp.string(), "val", /*imgsz=*/320,
                                       names, aug);
  EXPECT(val.size() == 1, "val split should have 1 image");
  auto vex = val.get(0);
  EXPECT(vex.targets.size(0) == 1, "val image should have 1 label");

  // Batch sanity.
  std::mt19937 rng(7);
  auto batch = train.sample_batch(/*bsz=*/4, rng);
  EXPECT(batch.imgs.dim() == 4 && batch.imgs.size(0) == 4,
         "batch.imgs should be [B,3,H,W] with B=4");
  EXPECT(batch.targets.size(1) == 6,
         "batch.targets should have 6 columns (batch_idx + 5)");

  // Bad-split error path.
  bool threw = false;
  try {
    yolocpp::datasets::FlatDataset bad(tmp.string(), "test", 320, names);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT(threw, "non-existent split should throw");

  fs::remove(tmp);
  std::cout << "[flat] OK — split filter, getitem, batch, error path\n";
  return 0;
}
