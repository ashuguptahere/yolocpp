// Smokes for VocDataset + CocoDataset (#54B). Each test synthesises
// a tiny dataset on the fly + verifies parsing + getitem returns
// well-shaped (img, targets).

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

#include <opencv2/imgcodecs.hpp>

#include "yolocpp/datasets/coco_dataset.hpp"
#include "yolocpp/datasets/voc_dataset.hpp"

#define EXPECT(cond, msg) \
  do {                    \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

namespace fs = std::filesystem;

int test_voc() {
  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[voc] SKIP — data/bus.jpg missing\n";
    return 0;
  }
  // Build a tiny VOC layout under /tmp.
  fs::path root = "/tmp/yolocpp_voc_test";
  fs::remove_all(root);
  fs::create_directories(root / "JPEGImages");
  fs::create_directories(root / "Annotations");
  fs::create_directories(root / "ImageSets" / "Main");
  fs::copy_file("data/bus.jpg", root / "JPEGImages" / "0001.jpg");

  // Read bus.jpg dims to write a real bbox.
  cv::Mat probe = cv::imread((root / "JPEGImages" / "0001.jpg").string());
  EXPECT(!probe.empty(), "test fixture: bus.jpg unreadable");
  int W = probe.cols, H = probe.rows;
  (void)W; (void)H;

  // Two boxes: a "person" + a "bus", in absolute pixel coords.
  std::ofstream(root / "Annotations" / "0001.xml") <<
      "<annotation><filename>0001.jpg</filename>"
      "<size><width>" << W << "</width><height>" << H << "</height>"
      "<depth>3</depth></size>"
      "<object><name>person</name>"
      "<bndbox><xmin>50</xmin><ymin>100</ymin>"
      "<xmax>200</xmax><ymax>500</ymax></bndbox></object>"
      "<object><name>bus</name>"
      "<bndbox><xmin>10</xmin><ymin>200</ymin>"
      "<xmax>" << (W - 10) << "</xmax><ymax>" << (H - 50) << "</ymax></bndbox></object>"
      "<object><name>UNKNOWN_CLASS_SHOULD_BE_SKIPPED</name>"
      "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>"
      "</annotation>\n";
  std::ofstream(root / "ImageSets" / "Main" / "train.txt") << "0001\n";

  yolocpp::datasets::AugConfig aug; aug.augment = false;
  yolocpp::datasets::VocDataset ds(root.string(), "train", /*imgsz=*/320,
                                     yolocpp::datasets::voc_default_names(),
                                     aug);
  EXPECT(ds.size() == 1, "voc: 1 image expected");
  auto ex = ds.get(0);
  EXPECT(ex.targets.size(0) == 2,
         "voc: 2 valid labels expected (unknown class skipped)");
  // person = idx 14, bus = idx 5 in voc_default_names().
  auto a = ex.targets.accessor<float, 2>();
  std::set<int> got;
  for (int i = 0; i < (int)ex.targets.size(0); ++i) got.insert((int)a[i][0]);
  EXPECT(got.count(14) == 1, "voc: 'person' should map to idx 14");
  EXPECT(got.count(5)  == 1, "voc: 'bus' should map to idx 5");

  fs::remove_all(root);
  std::cout << "[voc] OK\n";
  return 0;
}

int test_coco() {
  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[coco] SKIP — data/bus.jpg missing\n";
    return 0;
  }
  fs::path root = "/tmp/yolocpp_coco_test";
  fs::remove_all(root);
  fs::create_directories(root / "imgs");
  fs::copy_file("data/bus.jpg", root / "imgs" / "img.jpg");

  cv::Mat probe = cv::imread((root / "imgs" / "img.jpg").string());
  int W = probe.cols, H = probe.rows;

  fs::path j = root / "ann.json";
  std::ofstream(j) <<
      "{"
      "\"images\":[{\"id\":1,\"file_name\":\"img.jpg\","
                   "\"width\":" << W << ",\"height\":" << H << "}],"
      "\"annotations\":["
        "{\"image_id\":1,\"category_id\":1,\"bbox\":[10,20,80,150],\"id\":1},"
        "{\"image_id\":1,\"category_id\":7,\"bbox\":[100,30,200,400],\"id\":2}"
      "],"
      "\"categories\":["
        "{\"id\":1,\"name\":\"person\"},"
        "{\"id\":7,\"name\":\"bus\"}"
      "]"
      "}\n";

  yolocpp::datasets::AugConfig aug; aug.augment = false;
  yolocpp::datasets::CocoDataset ds(j.string(),
                                     (root / "imgs").string(),
                                     /*imgsz=*/320, aug);
  EXPECT(ds.size() == 1, "coco: 1 image expected");
  EXPECT(ds.num_classes() == 2, "coco: 2 categories expected");
  EXPECT(ds.names()[0] == "person" && ds.names()[1] == "bus",
         "coco: dense category remap should preserve order");
  auto ex = ds.get(0);
  EXPECT(ex.targets.size(0) == 2, "coco: 2 labels expected");
  auto a = ex.targets.accessor<float, 2>();
  std::set<int> got;
  for (int i = 0; i < (int)ex.targets.size(0); ++i) got.insert((int)a[i][0]);
  EXPECT(got.count(0) == 1 && got.count(1) == 1,
         "coco: cat IDs 1,7 should remap to 0,1");

  fs::remove_all(root);
  std::cout << "[coco] OK\n";
  return 0;
}

int main() {
  if (int rc = test_voc();  rc != 0) return rc;
  if (int rc = test_coco(); rc != 0) return rc;
  std::cout << "[voc+coco] all OK\n";
  return 0;
}
