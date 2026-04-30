#include "yolocpp/datasets/yolo_dataset.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "yolocpp/inference/letterbox.hpp"

namespace yolocpp::datasets {

namespace fs = std::filesystem;

namespace {

std::vector<std::string> list_images(const fs::path& dir) {
  std::vector<std::string> out;
  if (!fs::exists(dir)) return out;
  for (auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
      out.push_back(e.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string label_for_image(const std::string& img_path,
                            const fs::path& root) {
  // Replace .../images/.../X.jpg with .../labels/.../X.txt
  fs::path p(img_path);
  std::string s = img_path;
  auto pos = s.rfind("/images/");
  if (pos != std::string::npos) {
    s.replace(pos, 8, "/labels/");
  }
  fs::path lp(s);
  lp.replace_extension(".txt");
  (void)root;
  return lp.string();
}

torch::Tensor load_targets(const std::string& lbl_path) {
  std::ifstream f(lbl_path);
  if (!f.is_open()) return torch::zeros({0, 5}, torch::kFloat32);
  std::vector<float> rows;
  std::string line;
  int n_rows = 0;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int   cls;
    float cx, cy, w, h;
    if (!(iss >> cls >> cx >> cy >> w >> h)) continue;
    rows.push_back((float)cls);
    rows.push_back(cx); rows.push_back(cy);
    rows.push_back(w);  rows.push_back(h);
    ++n_rows;
  }
  return torch::from_blob(rows.data(), {n_rows, 5}, torch::kFloat32).clone();
}

void hsv_jitter(cv::Mat& bgr, std::mt19937& rng,
                float hg, float sg, float vg) {
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  float r_h = 1.f + u(rng) * hg;
  float r_s = 1.f + u(rng) * sg;
  float r_v = 1.f + u(rng) * vg;

  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  // OpenCV H is 0..179, S/V are 0..255
  ch[0].convertTo(ch[0], CV_32F);
  ch[1].convertTo(ch[1], CV_32F);
  ch[2].convertTo(ch[2], CV_32F);
  ch[0] = cv::min(cv::max(ch[0] * r_h, 0.f), 179.f);
  ch[1] = cv::min(cv::max(ch[1] * r_s, 0.f), 255.f);
  ch[2] = cv::min(cv::max(ch[2] * r_v, 0.f), 255.f);
  ch[0].convertTo(ch[0], CV_8U);
  ch[1].convertTo(ch[1], CV_8U);
  ch[2].convertTo(ch[2], CV_8U);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

}  // anonymous namespace

YoloDataset::YoloDataset(std::string root, std::string split, int imgsz,
                         std::vector<std::string> names, AugConfig aug)
    : imgsz_(imgsz), names_(std::move(names)), aug_(aug) {
  fs::path img_dir = fs::path(root) / "images" / split;
  fs::path lbl_dir = fs::path(root) / "labels" / split;

  img_paths_ = list_images(img_dir);
  lbl_paths_.reserve(img_paths_.size());
  for (auto& ip : img_paths_)
    lbl_paths_.push_back(label_for_image(ip, root));

  if (img_paths_.empty()) {
    throw std::runtime_error(
        "no images found at " + img_dir.string() +
        " (expected YOLO layout: <root>/images/<split>/*.jpg)");
  }
}

YoloDataset::YoloDataset(std::vector<std::string> img_paths,
                         std::vector<std::string> lbl_paths,
                         int imgsz, std::vector<std::string> names,
                         AugConfig aug)
    : img_paths_(std::move(img_paths)),
      lbl_paths_(std::move(lbl_paths)),
      imgsz_(imgsz),
      names_(std::move(names)),
      aug_(aug) {
  if (img_paths_.size() != lbl_paths_.size())
    throw std::runtime_error("YoloDataset: image/label list size mismatch");
}

YoloExample YoloDataset::get(std::size_t idx, uint64_t aug_seed) const {
  YoloExample ex;
  ex.img_path = img_paths_[idx];

  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load image: " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;

  std::mt19937 rng(aug_seed ? aug_seed
                            : (uint64_t)idx * 0x9E3779B97F4A7C15ULL);

  // Augmentation (color) before letterbox.
  if (aug_.augment) {
    hsv_jitter(bgr, rng, aug_.hsv_h, aug_.hsv_s, aug_.hsv_v);
  }

  // Letterbox. `rect=true` (val convention) pads only to a multiple of 32
  // instead of squaring to imgsz×imgsz — fewer dead pixels, matches
  // Ultralytics' val mAP path.
  auto lb = inference::letterbox(bgr, imgsz_,
                                  /*pad_color=*/cv::Scalar(114, 114, 114),
                                  /*scale_up=*/false,
                                  /*auto_minrec=*/aug_.rect);
  cv::Mat lb_img = lb.img;
  ex.gain  = lb.gain;
  ex.pad_x = lb.pad_x;
  ex.pad_y = lb.pad_y;

  // Optional horizontal flip.
  bool flip = false;
  if (aug_.augment) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < aug_.flip_p;
    if (flip) cv::flip(lb_img, lb_img, /*flipCode=*/1);  // horizontal
  }

  ex.img = inference::image_to_tensor(lb_img);

  // Targets: read normalized YOLO labels and map into letterboxed pixel coords.
  auto labels = load_targets(lbl_paths_[idx]);  // [N, 5] (cls, cx, cy, w, h) ∈ [0,1]
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

YoloDataset::Batch YoloDataset::sample_batch(std::size_t bsz,
                                             std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, img_paths_.size() - 1);
  std::vector<YoloExample> exs;
  exs.reserve(bsz);
  std::vector<torch::Tensor> imgs;
  std::vector<torch::Tensor> tgts_with_b;

  for (size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), /*aug_seed=*/((uint64_t)rng()) << 32 | i);
    auto t  = ex.targets;  // [N, 5]
    imgs.push_back(ex.img);
    if (t.size(0) > 0) {
      auto bcol = torch::full({t.size(0), 1}, (double)i, torch::kFloat32);
      tgts_with_b.push_back(torch::cat({bcol, t}, /*dim=*/1));
    }
    exs.push_back(std::move(ex));
  }
  Batch out;
  out.imgs = torch::stack(imgs, /*dim=*/0);  // [B, 3, H, W]
  out.targets = tgts_with_b.empty()
                  ? torch::zeros({0, 6}, torch::kFloat32)
                  : torch::cat(tgts_with_b, /*dim=*/0);
  out.examples = std::move(exs);
  return out;
}

}  // namespace yolocpp::datasets
