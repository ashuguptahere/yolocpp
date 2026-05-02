// Pascal VOC dataset (#54B). Reads ImageSets/Main/<split>.txt for
// the image-id list, parses Annotations/<id>.xml for bboxes, looks
// up class names against the provided `names`. Stores per-image
// label tensors in YOLO-normalised (cls, cx, cy, w, h) form so
// getitem() shares the same letterbox + augmentation pipeline as
// the YOLO and Flat datasets.
//
// XML parsing is intentionally minimal — Pascal VOC's annotation
// schema is fixed (every fork copies it verbatim) and we only care
// about <name>, <bndbox>, <xmin/ymin/xmax/ymax>. A from-scratch
// regex extractor saves us from pulling in libxml2.

#include "yolocpp/datasets/voc_dataset.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolocpp/inference/letterbox.hpp"

namespace fs = std::filesystem;

namespace yolocpp::datasets {

const std::vector<std::string>& voc_default_names() {
  static const std::vector<std::string> N = {
      "aeroplane", "bicycle", "bird", "boat",     "bottle",
      "bus",       "car",     "cat",  "chair",    "cow",
      "diningtable","dog",    "horse","motorbike","person",
      "pottedplant","sheep",  "sofa", "train",    "tvmonitor",
  };
  return N;
}

namespace {

// hsv_jitter: shared with YoloDataset / FlatDataset. Pending #54E
// dataset-loader factoring; duplicating ~30 lines is fine for now.
void hsv_jitter(cv::Mat& bgr, std::mt19937& rng,
                float h_amp, float s_amp, float v_amp) {
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  float h = u(rng) * h_amp;
  float s = u(rng) * s_amp;
  float v = u(rng) * v_amp;
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch; cv::split(hsv, ch);
  ch[0].convertTo(ch[0], CV_32F);
  ch[1].convertTo(ch[1], CV_32F);
  ch[2].convertTo(ch[2], CV_32F);
  ch[0] = (ch[0] + h * 180.f);
  for (int r = 0; r < ch[0].rows; ++r) {
    auto* p = ch[0].ptr<float>(r);
    for (int c = 0; c < ch[0].cols; ++c) {
      float x = p[c]; while (x < 0)   x += 180.f;
                       while (x >= 180) x -= 180.f;
      p[c] = x;
    }
  }
  ch[1] = cv::min(cv::max(ch[1] * (1.f + s), 0.f), 255.f);
  ch[2] = cv::min(cv::max(ch[2] * (1.f + v), 0.f), 255.f);
  ch[0].convertTo(ch[0], CV_8U);
  ch[1].convertTo(ch[1], CV_8U);
  ch[2].convertTo(ch[2], CV_8U);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

// Extract every `<object>` block and return its (name, xmin, ymin,
// xmax, ymax). Ignores difficult / truncated flags — VOC train/val
// use them inconsistently and downstream pipelines mostly include
// difficult=1 boxes anyway.
struct VocBox { std::string name; float x1, y1, x2, y2; };

std::vector<VocBox> parse_voc_xml(const std::string& xml_path) {
  std::ifstream f(xml_path);
  if (!f.is_open()) {
    throw std::runtime_error("VocDataset: cannot open xml: " + xml_path);
  }
  std::stringstream ss; ss << f.rdbuf();
  std::string body = ss.str();

  // Split on </object> so each chunk contains one <object>'s fields
  // (or none, for the trailing chunk).
  std::vector<VocBox> out;
  static const std::regex obj_re(R"(<object>([\s\S]*?)</object>)");
  static const std::regex name_re(R"(<name>\s*([^<\s][^<]*?)\s*</name>)");
  static const std::regex x1_re(R"(<xmin>\s*([\d\.\-]+)\s*</xmin>)");
  static const std::regex y1_re(R"(<ymin>\s*([\d\.\-]+)\s*</ymin>)");
  static const std::regex x2_re(R"(<xmax>\s*([\d\.\-]+)\s*</xmax>)");
  static const std::regex y2_re(R"(<ymax>\s*([\d\.\-]+)\s*</ymax>)");

  for (auto it = std::sregex_iterator(body.begin(), body.end(), obj_re);
       it != std::sregex_iterator(); ++it) {
    std::string blk = (*it)[1].str();
    std::smatch m;
    VocBox b{};
    if (!std::regex_search(blk, m, name_re)) continue;
    b.name = m[1];
    if (!std::regex_search(blk, m, x1_re)) continue; b.x1 = std::stof(m[1]);
    if (!std::regex_search(blk, m, y1_re)) continue; b.y1 = std::stof(m[1]);
    if (!std::regex_search(blk, m, x2_re)) continue; b.x2 = std::stof(m[1]);
    if (!std::regex_search(blk, m, y2_re)) continue; b.y2 = std::stof(m[1]);
    out.push_back(std::move(b));
  }
  return out;
}

}  // namespace

VocDataset::VocDataset(std::string root, std::string split, int imgsz,
                       std::vector<std::string> names, AugConfig aug)
    : imgsz_(imgsz), names_(std::move(names)), aug_(std::move(aug)) {
  fs::path r(root);
  fs::path imgsets = r / "ImageSets" / "Main" / (split + ".txt");
  fs::path img_dir = r / "JPEGImages";
  fs::path ann_dir = r / "Annotations";

  if (!fs::exists(imgsets))
    throw std::runtime_error("VocDataset: missing " + imgsets.string());
  if (!fs::is_directory(img_dir))
    throw std::runtime_error("VocDataset: missing " + img_dir.string());
  if (!fs::is_directory(ann_dir))
    throw std::runtime_error("VocDataset: missing " + ann_dir.string());

  // class-name → idx map.
  std::unordered_map<std::string, int> name_to_idx;
  for (int i = 0; i < (int)names_.size(); ++i) name_to_idx[names_[i]] = i;

  std::ifstream f(imgsets);
  std::string id;
  while (std::getline(f, id)) {
    while (!id.empty() && std::isspace((unsigned char)id.back())) id.pop_back();
    if (id.empty() || id[0] == '#') continue;

    fs::path img_p = img_dir / (id + ".jpg");
    fs::path xml_p = ann_dir / (id + ".xml");
    if (!fs::exists(img_p)) {
      // VOC has occasional .png too; tolerate both.
      auto alt = img_dir / (id + ".png");
      if (fs::exists(alt)) img_p = alt;
      else continue;
    }
    if (!fs::exists(xml_p)) continue;

    cv::Mat probe = cv::imread(img_p.string(), cv::IMREAD_REDUCED_COLOR_8);
    if (probe.empty()) continue;
    int W = probe.cols * 8;  // approximate; we'll use the actual image dims at getitem
    int H = probe.rows * 8;
    // Use the precise dims via imread metadata: re-read header.
    {
      cv::Mat actual = cv::imread(img_p.string());
      if (!actual.empty()) { W = actual.cols; H = actual.rows; }
    }

    auto boxes = parse_voc_xml(xml_p.string());
    std::vector<float> rows;
    int n_kept = 0;
    for (const auto& b : boxes) {
      auto it = name_to_idx.find(b.name);
      if (it == name_to_idx.end()) {
        // Unknown class — skip silently. Pass a custom `names` if
        // your VOC fork has extra classes.
        continue;
      }
      // VOC bboxes are 1-indexed pixel coords; we treat them as
      // 0-indexed and convert to YOLO-normalised (cx, cy, w, h).
      float x1 = std::max(0.f, b.x1 - 1.f);
      float y1 = std::max(0.f, b.y1 - 1.f);
      float x2 = std::min((float)W, b.x2 - 1.f);
      float y2 = std::min((float)H, b.y2 - 1.f);
      if (x2 <= x1 || y2 <= y1) continue;
      float cx = ((x1 + x2) * 0.5f) / (float)W;
      float cy = ((y1 + y2) * 0.5f) / (float)H;
      float w  = (x2 - x1) / (float)W;
      float h  = (y2 - y1) / (float)H;
      rows.push_back((float)it->second);
      rows.push_back(cx); rows.push_back(cy);
      rows.push_back(w);  rows.push_back(h);
      ++n_kept;
    }
    img_paths_.push_back(img_p.string());
    if (n_kept == 0) {
      labels_.push_back(torch::zeros({0, 5}, torch::kFloat32));
    } else {
      labels_.push_back(
          torch::from_blob(rows.data(), {n_kept, 5}, torch::kFloat32).clone());
    }
  }

  if (img_paths_.empty())
    throw std::runtime_error(
        "VocDataset: no images for split='" + split + "' under " + root);

  std::cout << "[voc] " << root << " split=" << split
            << ": " << img_paths_.size() << " images, "
            << [&]() {
                 std::size_t t = 0;
                 for (auto& l : labels_) t += (std::size_t)l.size(0);
                 return t;
               }()
            << " labels (over " << names_.size() << " classes)\n";
}

YoloExample VocDataset::get(std::size_t idx, std::uint64_t aug_seed) const {
  YoloExample ex;
  ex.img_path = img_paths_[idx];

  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("VocDataset: failed to load " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;

  std::mt19937 rng(aug_seed ? aug_seed
                              : (std::uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  if (aug_.augment) {
    hsv_jitter(bgr, rng, aug_.hsv_h, aug_.hsv_s, aug_.hsv_v);
  }

  auto lb = inference::letterbox(bgr, imgsz_,
                                  /*pad_color=*/cv::Scalar(114, 114, 114),
                                  /*scale_up=*/false,
                                  /*auto_minrec=*/aug_.rect);
  cv::Mat lb_img = lb.img;
  ex.gain  = lb.gain;
  ex.pad_x = lb.pad_x;
  ex.pad_y = lb.pad_y;

  bool flip = false;
  if (aug_.augment) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < aug_.flip_p;
    if (flip) cv::flip(lb_img, lb_img, /*flipCode=*/1);
  }
  ex.img = inference::image_to_tensor(lb_img);

  const auto& labels = labels_[idx];
  if (labels.size(0) == 0) {
    ex.targets = torch::zeros({0, 5}, torch::kFloat32);
    return ex;
  }
  auto t = labels.clone();
  auto a = t.accessor<float, 2>();
  for (int i = 0; i < (int)t.size(0); ++i) {
    float cx = a[i][1] * (float)ex.orig_w;
    float cy = a[i][2] * (float)ex.orig_h;
    float w  = a[i][3] * (float)ex.orig_w;
    float h  = a[i][4] * (float)ex.orig_h;
    cx = (float)(cx * lb.gain + lb.pad_x);
    cy = (float)(cy * lb.gain + lb.pad_y);
    w  = (float)(w  * lb.gain);
    h  = (float)(h  * lb.gain);
    if (flip) cx = (float)imgsz_ - 1.f - cx;
    a[i][1] = cx; a[i][2] = cy; a[i][3] = w; a[i][4] = h;
  }
  ex.targets = t;
  return ex;
}

VocDataset::Batch VocDataset::sample_batch(std::size_t bsz,
                                            std::mt19937& rng) const {
  std::uniform_int_distribution<std::size_t> u(0, img_paths_.size() - 1);
  Batch b;
  std::vector<torch::Tensor> imgs, tgts_with_b;
  for (std::size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), /*aug_seed=*/((std::uint64_t)rng()) << 32 | i);
    imgs.push_back(ex.img);
    if (ex.targets.size(0) > 0) {
      auto bcol = torch::full({ex.targets.size(0), 1}, (int64_t)i,
                                torch::kFloat32);
      tgts_with_b.push_back(torch::cat({bcol, ex.targets}, /*dim=*/1));
    }
    b.examples.push_back(std::move(ex));
  }
  b.imgs = torch::stack(imgs, /*dim=*/0);
  b.targets = tgts_with_b.empty()
                  ? torch::zeros({0, 6}, torch::kFloat32)
                  : torch::cat(tgts_with_b, /*dim=*/0);
  return b;
}

}  // namespace yolocpp::datasets
