// Implementation of `FlatDataset` (#54A). Parses a single-file
// CSV/TSV holding (split, image_path, cls, cx, cy, w, h) rows;
// groups by image; produces standard YoloExample for getitem.
//
// The image-loading + letterbox + augmentation pipeline mirrors
// `YoloDataset::get` so trainer + validator can consume either
// dataset interchangeably. The duplication is intentional (one
// session of dataset-loader factoring lives under #54E); we
// optimise for "easy to read in isolation" until the unification.

#include "yolocpp/datasets/flat_dataset.hpp"

#include "yolocpp/datasets/augment.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolocpp/inference/letterbox.hpp"

namespace fs = std::filesystem;

namespace yolocpp::datasets {

namespace {

// Trim leading/trailing whitespace.
std::string trim(std::string s) {
  while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(0, 1);
  while (!s.empty() && std::isspace((unsigned char)s.back()))  s.pop_back();
  return s;
}

// Pick the field separator from the header. The header MUST contain
// `split` and `image_path` in some order; whichever delimiter splits
// it into the right token count (≥ 7) wins. Falls back to comma.
char detect_delim(const std::string& header) {
  for (char c : {'\t', ',', ';'}) {
    int n = 1;
    for (char ch : header) if (ch == c) ++n;
    if (n >= 7) return c;
  }
  return ',';
}

// Tokenise `line` on `delim`, preserving empty tokens (so optional
// label columns can be empty).
std::vector<std::string> split_line(const std::string& line, char delim) {
  std::vector<std::string> out;
  std::string buf;
  for (char c : line) {
    if (c == delim) { out.push_back(buf); buf.clear(); }
    else            { buf += c; }
  }
  out.push_back(buf);
  return out;
}

// hsv_jitter now lives in datasets/augment.{hpp,cpp} (#54E) — the shared
// LUT-based form, parity-correct with YoloDataset (replaces the old
// float-domain per-pixel wrap loop).

}  // namespace

FlatDataset::FlatDataset(std::string file_path, std::string split, int imgsz,
                         std::vector<std::string> names, AugConfig aug,
                         std::uint64_t seed)
    : imgsz_(imgsz), names_(std::move(names)), aug_(std::move(aug)),
      seed_(seed) {
  if (!fs::exists(file_path))
    throw std::runtime_error("FlatDataset: file not found: " + file_path);

  std::ifstream f(file_path);
  if (!f.is_open())
    throw std::runtime_error("FlatDataset: cannot open: " + file_path);

  std::string header;
  if (!std::getline(f, header))
    throw std::runtime_error("FlatDataset: empty file: " + file_path);

  char delim = detect_delim(header);
  auto cols = split_line(header, delim);
  for (auto& c : cols) c = trim(c);

  // Build a column-name → index map. Required: split, image_path,
  // class_id, x_center, y_center, width, height. Names are
  // case-insensitive; `cls` accepted as an alias for class_id.
  auto lower = [](std::string s) {
    for (auto& c : s) c = std::tolower((unsigned char)c);
    return s;
  };
  int  i_split = -1, i_img = -1, i_cls = -1;
  int  i_cx = -1, i_cy = -1, i_w = -1, i_h = -1;
  for (int i = 0; i < (int)cols.size(); ++i) {
    auto n = lower(cols[i]);
    if      (n == "split")              i_split = i;
    else if (n == "image_path" ||
             n == "image" ||
             n == "img")                i_img = i;
    else if (n == "class_id" || n == "cls" || n == "class")
                                          i_cls = i;
    else if (n == "x_center" || n == "cx") i_cx = i;
    else if (n == "y_center" || n == "cy") i_cy = i;
    else if (n == "width"    || n == "w")  i_w  = i;
    else if (n == "height"   || n == "h")  i_h  = i;
  }
  if (i_split < 0 || i_img < 0)
    throw std::runtime_error(
        "FlatDataset: header must contain 'split' and 'image_path' columns");
  if (i_cls < 0 || i_cx < 0 || i_cy < 0 || i_w < 0 || i_h < 0)
    throw std::runtime_error(
        "FlatDataset: header must contain class_id, x_center, y_center, "
        "width, height columns");

  // image_path resolution: relative paths are relative to the
  // dataset file's directory.
  fs::path base_dir = fs::path(file_path).parent_path();

  // Group rows by image. We keep insertion order (first appearance
  // of an image_path) so the dataset is deterministic across runs.
  std::vector<std::string>           ordered_imgs;
  std::vector<std::vector<float>>    flat_labels;  // 5 floats per label
  std::unordered_map<std::string, std::size_t> idx_of_img;

  std::string line;
  std::size_t lineno = 1;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    auto tok = split_line(line, delim);
    if ((int)tok.size() <= std::max({i_split, i_img, i_cls, i_cx, i_cy, i_w, i_h})) {
      // Tolerate ragged lines; just skip them with a warning the
      // first time. Don't spam.
      static bool warned = false;
      if (!warned) {
        std::cerr << "[flat] line " << lineno
                  << ": fewer columns than header expected — skipping (further warnings suppressed)\n";
        warned = true;
      }
      continue;
    }
    auto sp = trim(tok[i_split]);
    if (sp != split) continue;

    auto rel = trim(tok[i_img]);
    fs::path img_p(rel);
    if (img_p.is_relative()) img_p = base_dir / img_p;
    auto img_s = img_p.lexically_normal().string();

    std::size_t img_idx;
    auto it = idx_of_img.find(img_s);
    if (it == idx_of_img.end()) {
      img_idx = ordered_imgs.size();
      idx_of_img[img_s] = img_idx;
      ordered_imgs.push_back(img_s);
      flat_labels.emplace_back();
    } else {
      img_idx = it->second;
    }

    auto cls_s = trim(tok[i_cls]);
    if (cls_s.empty()) continue;  // background-only row for this image

    try {
      int   cls = std::stoi(cls_s);
      float cx  = std::stof(trim(tok[i_cx]));
      float cy  = std::stof(trim(tok[i_cy]));
      float w   = std::stof(trim(tok[i_w]));
      float h   = std::stof(trim(tok[i_h]));
      auto& v = flat_labels[img_idx];
      v.push_back((float)cls);
      v.push_back(cx); v.push_back(cy);
      v.push_back(w);  v.push_back(h);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "FlatDataset: line " + std::to_string(lineno) +
          ": malformed numeric field (" + e.what() + ")");
    }
  }

  if (ordered_imgs.empty())
    throw std::runtime_error(
        "FlatDataset: no rows match split='" + split + "' in " + file_path);

  // Materialise per-image label tensors.
  img_paths_ = std::move(ordered_imgs);
  labels_.reserve(img_paths_.size());
  for (auto& v : flat_labels) {
    int n = (int)(v.size() / 5);
    if (n == 0) {
      labels_.push_back(torch::zeros({0, 5}, torch::kFloat32));
    } else {
      labels_.push_back(
          torch::from_blob(v.data(), {n, 5}, torch::kFloat32).clone());
    }
  }

  std::cout << "[flat] " << file_path << " split=" << split
            << ": " << img_paths_.size() << " images, "
            << [&]() {
                 std::size_t t = 0;
                 for (auto& l : labels_) t += (std::size_t)l.size(0);
                 return t;
               }()
            << " labels\n";
}

YoloExample FlatDataset::get(std::size_t idx, std::uint64_t aug_seed) const {
  YoloExample ex;
  ex.img_path = img_paths_[idx];

  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("FlatDataset: failed to load image: " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;

  // Per-call RNG: use the explicit aug_seed if given; else fold in
  // the dataset-level seed (0 ⇒ same per-idx behaviour as YoloDataset).
  std::uint64_t s = aug_seed ? aug_seed
                              : ((std::uint64_t)idx * 0x9E3779B97F4A7C15ULL ^ seed_);
  std::mt19937 rng(s);

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
    if (flip) cx = (float)imgsz_ - cx;  // w - cx (not w-1), match YoloDataset+upstream
    a[i][1] = cx; a[i][2] = cy; a[i][3] = w; a[i][4] = h;
  }
  ex.targets = t;
  return ex;
}

FlatDataset::Batch FlatDataset::sample_batch(std::size_t bsz,
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
