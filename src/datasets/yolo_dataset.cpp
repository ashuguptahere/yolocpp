#include "yolocpp/datasets/yolo_dataset.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

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
  // Matches ultralytics RandomHSV exactly (data/augment.py:1463):
  //   r = uniform(-1, 1) * [hgain, sgain, vgain]
  //   lut_hue = ((x + r[0] * 180) % 180)         ← ADDITIVE + WRAP
  //   lut_sat = clip(x * (r[1] + 1), 0, 255)     ← multiplicative + clip
  //   lut_val = clip(x * (r[2] + 1), 0, 255)     ← multiplicative + clip
  //   lut_sat[0] = 0   (8.3.79+: keep pure white pure)
  //
  // Pre-0.99.0 yolocpp used `hue *= r_h` with clip — that's
  // multiplicative-non-wrapping, which biases hues toward 0/179
  // (red/violet) and produces a different aug distribution than
  // upstream. The fix is the LUT-with-wrap approach.
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const float r_h = u(rng) * hg;
  const float r_s = u(rng) * sg;
  const float r_v = u(rng) * vg;

  // Build the three LUTs to match upstream exactly.
  std::array<uint8_t, 256> lut_h{}, lut_s{}, lut_v{};
  const float hue_off = r_h * 180.f;
  for (int x = 0; x < 256; ++x) {
    // Hue: additive offset, wrap mod 180. `% 180` in numpy on a
    // negative integer returns non-negative; emulate that with
    // `((a % 180) + 180) % 180`.
    int hv = (int)std::lround((float)x + hue_off) % 180;
    if (hv < 0) hv += 180;
    lut_h[x] = (uint8_t)hv;
    // Sat / Val: multiplicative gain (r+1) with hard clip.
    int sv = (int)std::lround((float)x * (r_s + 1.f));
    int vv = (int)std::lround((float)x * (r_v + 1.f));
    lut_s[x] = (uint8_t)std::clamp(sv, 0, 255);
    lut_v[x] = (uint8_t)std::clamp(vv, 0, 255);
  }
  lut_s[0] = 0;  // 8.3.79+ rule — preserve pure white

  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  cv::Mat lut_h_mat(1, 256, CV_8UC1, lut_h.data());
  cv::Mat lut_s_mat(1, 256, CV_8UC1, lut_s.data());
  cv::Mat lut_v_mat(1, 256, CV_8UC1, lut_v.data());
  cv::LUT(ch[0], lut_h_mat, ch[0]);
  cv::LUT(ch[1], lut_s_mat, ch[1]);
  cv::LUT(ch[2], lut_v_mat, ch[2]);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

}  // anonymous namespace

// Pre-decode all images into cache_imgs_ (one-shot at ctor time when
// aug.cache_ram is set). Decoding 2465 JPEGs in parallel on 16 cores
// takes ~3-5 s on the screen dataset. After this, ds.get() and
// build_mosaic4 never touch the disk again — the dominant CPU cost
// on small-model training disappears.
//
// Static-private free function (the lambda is std::thread-friendly).
static void populate_image_cache(
    const std::vector<std::string>& img_paths,
    std::vector<cv::Mat>& cache,
    int num_threads = 8) {
  cache.assign(img_paths.size(), cv::Mat{});
  std::atomic<std::size_t> next{0};
  auto worker = [&] {
    while (true) {
      std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= img_paths.size()) return;
      cv::Mat bgr = cv::imread(img_paths[i], cv::IMREAD_COLOR);
      if (bgr.empty()) {
        throw std::runtime_error("cache_ram: failed to decode " + img_paths[i]);
      }
      cache[i] = std::move(bgr);
    }
  };
  std::vector<std::thread> ts;
  ts.reserve((std::size_t)num_threads);
  for (int t = 0; t < num_threads; ++t) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
}

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
  if (aug_.cache_ram) populate_image_cache(img_paths_, cache_imgs_);
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
  if (aug_.cache_ram) populate_image_cache(img_paths_, cache_imgs_);
}

// Pre-loaded ctor: COCO JSON / Pascal VOC / Flat CSV all parse into
// (img_paths, per-image [N,5] label tensors), then funnel through
// here so downstream trainer + validator can stay typed on
// `YoloDataset`. `lbl_paths_` is filled with empty strings of the
// matching length so any size-check elsewhere still works.
YoloDataset::YoloDataset(std::vector<std::string> img_paths,
                          std::vector<torch::Tensor> labels,
                          int imgsz, std::vector<std::string> names,
                          AugConfig aug)
    : img_paths_(std::move(img_paths)),
      lbl_paths_(),
      pre_labels_(std::move(labels)),
      imgsz_(imgsz),
      names_(std::move(names)),
      aug_(std::move(aug)) {
  if (img_paths_.size() != pre_labels_.size())
    throw std::runtime_error(
        "YoloDataset(pre-loaded): img_paths/labels size mismatch");
  lbl_paths_.assign(img_paths_.size(), "");
  if (aug_.cache_ram) populate_image_cache(img_paths_, cache_imgs_);
}

YoloExample YoloDataset::get(std::size_t idx, uint64_t aug_seed) const {
  YoloExample ex;
  ex.img_path = img_paths_[idx];

  // RAM cache hit: clone so hsv_jitter can mutate without affecting
  // other workers reading the same cached entry concurrently.
  // Miss: fall back to cv::imread (lazy path; pre-0.98.0 behaviour).
  cv::Mat bgr;
  if (!cache_imgs_.empty()) {
    bgr = cache_imgs_[idx].clone();
  } else {
    bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  }
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
  // the upstream val mAP path.
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

  // Targets: prefer pre-loaded tensors when the dataset was built
  // from a non-YOLO format (#54B); otherwise parse the .txt file.
  auto labels = !pre_labels_.empty()
                  ? pre_labels_[idx]
                  : load_targets(lbl_paths_[idx]);
  // [N, 5] (cls, cx, cy, w, h) ∈ [0,1]
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

    // Bbox flip math matches Ultralytics' Instances.fliplr exactly:
    //   new_cx = w - old_cx   (NOT w - 1 - old_cx)
    // Reference: ultralytics/utils/instance.py:368 —
    //   self.bboxes[:, 0] = w - self.bboxes[:, 0]
    // The pre-0.99.0 yolocpp `w - 1 - cx` was off by 1 pixel; over
    // many flipped batches during training, accumulates as a slight
    // box-localization bias against upstream parity.
    if (flip) cx = (float)imgsz_ - cx;

    a[i][1] = cx; a[i][2] = cy; a[i][3] = w; a[i][4] = h;
  }
  ex.targets = t;
  return ex;
}

// Forward decl — defined below; build_mosaic4 calls it to apply the
// affine warp + s×s crop on the 2s mosaic canvas in one shot.
static std::pair<cv::Mat, std::vector<float>>
random_perspective_mat(const cv::Mat& in_img,
                        const std::vector<float>& in_lbls,
                        int out_w, int out_h,
                        const AugConfig& aug,
                        std::mt19937& rng);

// ─── Mosaic-4 (#54D) ────────────────────────────────────────────────────
// Build one example by stitching 4 sampled images into a 2*imgsz
// canvas with a random center, then applying RandomPerspective with
// output_size=(imgsz, imgsz) — that one call combines the warp with
// the crop, matching Ultralytics' pipeline order and avoiding the
// dead-pixel issue you get when stacking warp on an already-cropped
// mosaic.
//
// `anchor` is optional: when set (≥ 0), tile 0 is taken from that
// dataset index and tiles 1-3 are random. This lets the
// without-replacement BatchPrefetcher path guarantee every image
// becomes the anchor exactly once per epoch. anchor = -1 keeps the
// pre-0.94.x behaviour (all 4 tiles random).
static YoloExample build_mosaic4(const YoloDataset& ds, std::size_t imgsz,
                                  const AugConfig& aug,
                                  std::mt19937& rng,
                                  std::int64_t anchor = -1) {
  std::uniform_int_distribution<std::size_t> u_idx(0, ds.size() - 1);
  std::array<YoloExample, 4> tiles;
  for (int t = 0; t < 4; ++t) {
    if (t == 0 && anchor >= 0) {
      tiles[t] = ds.get((std::size_t)anchor);
    } else {
      tiles[t] = ds.get(u_idx(rng));
    }
  }

  int s = (int)imgsz;
  // Random mosaic center in [imgsz*0.5, imgsz*1.5] of the 2s canvas.
  std::uniform_int_distribution<int> u_center(s / 2, 3 * s / 2);
  int xc = u_center(rng);
  int yc = u_center(rng);

  // Build 2s × 2s canvas, fill with pad colour 114.
  cv::Mat canvas(2 * s, 2 * s, CV_8UC3, cv::Scalar(114, 114, 114));
  std::vector<float> all_labels;  // (cls, cx, cy, w, h) in mosaic-px coords

  for (int i = 0; i < 4; ++i) {
    // Convert tile.img tensor [3,H,W] back to BGR cv::Mat. Tiles
    // come from get() already letterboxed to imgsz×imgsz; we just
    // need raw pixels.
    auto t   = tiles[i].img.permute({1, 2, 0}).contiguous();
    cv::Mat tile_bgr(t.size(0), t.size(1), CV_32FC3, t.data_ptr());
    cv::Mat tile_u8;
    tile_bgr.convertTo(tile_u8, CV_8UC3, 255.0);
    cv::cvtColor(tile_u8, tile_u8, cv::COLOR_RGB2BGR);

    int tw = tile_u8.cols, th = tile_u8.rows;
    int x1a, y1a, x2a, y2a;  // mosaic-canvas ROI
    int x1b, y1b, x2b, y2b;  // tile ROI
    if (i == 0) {  // top-left
      x1a = std::max(xc - tw, 0);   y1a = std::max(yc - th, 0);
      x2a = xc;                     y2a = yc;
      x1b = tw - (x2a - x1a);       y1b = th - (y2a - y1a);
      x2b = tw;                     y2b = th;
    } else if (i == 1) {  // top-right
      x1a = xc;                     y1a = std::max(yc - th, 0);
      x2a = std::min(xc + tw, 2*s); y2a = yc;
      x1b = 0;                      y1b = th - (y2a - y1a);
      x2b = std::min(tw, x2a - x1a); y2b = th;
    } else if (i == 2) {  // bottom-left
      x1a = std::max(xc - tw, 0);   y1a = yc;
      x2a = xc;                     y2a = std::min(yc + th, 2*s);
      x1b = tw - (x2a - x1a);       y1b = 0;
      x2b = tw;                     y2b = std::min(th, y2a - y1a);
    } else {  // bottom-right
      x1a = xc;                     y1a = yc;
      x2a = std::min(xc + tw, 2*s); y2a = std::min(yc + th, 2*s);
      x1b = 0;                      y1b = 0;
      x2b = std::min(tw, x2a - x1a); y2b = std::min(th, y2a - y1a);
    }
    int dx = x1a - x1b;
    int dy = y1a - y1b;
    cv::Rect roi_a(x1a, y1a, x2a - x1a, y2a - y1a);
    cv::Rect roi_b(x1b, y1b, x2b - x1b, y2b - y1b);
    if (roi_a.width <= 0 || roi_a.height <= 0) continue;
    tile_u8(roi_b).copyTo(canvas(roi_a));

    // Translate this tile's bboxes (which are in tile-pixel coords
    // because YoloDataset::get returns pixel coords inside the
    // letterboxed image) by (dx, dy) into mosaic-canvas coords.
    if (tiles[i].targets.size(0) > 0) {
      auto a = tiles[i].targets.accessor<float, 2>();
      for (int k = 0; k < (int)tiles[i].targets.size(0); ++k) {
        all_labels.push_back(a[k][0]);
        all_labels.push_back(a[k][1] + dx);
        all_labels.push_back(a[k][2] + dy);
        all_labels.push_back(a[k][3]);
        all_labels.push_back(a[k][4]);
      }
    }
  }

  // Combined warp + crop via random_perspective_mat: input is the 2s
  // canvas with bboxes in 2s coords; output is the s×s region centred
  // on the canvas center with random affine applied. When the affine
  // knobs are all zero this collapses to a pure central crop (the
  // (xc, yc)-centred random crop variant is gone; spatial variability
  // now comes from the warp's translate). Matches Ultralytics' Mosaic
  // pipeline order: stitch 2s → random_perspective(out=s).
  auto [crop, kept] = random_perspective_mat(canvas, all_labels,
                                              s, s, aug, rng);

  YoloExample out;
  // image_to_tensor expects uint8 BGR (or BGR cv::Mat in either depth
  // — see letterbox.cpp). canvas was built from cv::COLOR_RGB2BGR'd
  // tiles so it's BGR.
  out.img = inference::image_to_tensor(crop);
  if (kept.empty()) {
    out.targets = torch::zeros({0, 5}, torch::kFloat32);
  } else {
    int n = (int)(kept.size() / 5);
    out.targets = torch::from_blob(kept.data(), {n, 5},
                                    torch::kFloat32).clone();
  }
  out.orig_w = s; out.orig_h = s; out.gain = 1.0;
  return out;
}

// RandomPerspective (#57G): affine warp (rotate + scale + translate)
// integrated into the mosaic pipeline per upstream Ultralytics. The
// warp is applied to the 2s mosaic canvas and the output is cropped
// to (out_w, out_h) centred on the input — this avoids the
// dead-pixel issue you get when you stack RandomPerspective on top
// of an already-cropped mosaic output. For non-mosaic samples
// (which are already letterboxed to imgsz×imgsz with no surrounding
// canvas to crop into) RandomPerspective is intentionally skipped.
//
// Affine matrix layout (2x3 for cv::warpAffine):
//   M = T_random · T_recenter · R(angle, scale)
//   where R is built via getRotationMatrix2D around the input
//   center, T_recenter shifts the warped center to (out_w/2, out_h/2),
//   and T_random adds the random translate.
// Bboxes (in input-coord space): 4 corners transformed by the same
// M; aabb; clip to (out_w, out_h); drop boxes with side < 2 px or
// post-warp area < 10% of original.
//
// in_img:  HWC float32 cv::Mat (any channel order — warpAffine is
//          channel-agnostic)
// in_lbls: flat (cls, cx, cy, w, h) flat array in input-coord pixels
// out_w/h: output canvas size (may differ from input — that's how we
//          combine warp + crop in one cv::warpAffine call)
//
// Returns: (warped HWC float32 cv::Mat, transformed labels) in
// output-coord pixels, ready to be packed into a YoloExample.
static std::pair<cv::Mat, std::vector<float>>
random_perspective_mat(const cv::Mat& in_img,
                        const std::vector<float>& in_lbls,
                        int out_w, int out_h,
                        const AugConfig& aug,
                        std::mt19937& rng) {
  const int W_in = in_img.cols, H_in = in_img.rows;
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  std::uniform_real_distribution<float> uscale(1.f - aug.scale_amp,
                                                1.f + aug.scale_amp);
  const double angle_deg = (double)(aug.degrees * u(rng));
  const double scale     = (double)uscale(rng);
  const double tx        = (double)(aug.translate * (float)out_w * u(rng));
  const double ty        = (double)(aug.translate * (float)out_h * u(rng));

  // Build affine: rotate+scale around input center, then shift so
  // the output's [0,0] maps to ((W_in - out_w)/2, (H_in - out_h)/2),
  // then add the random translate.
  cv::Mat M = cv::getRotationMatrix2D(
      cv::Point2d(W_in * 0.5, H_in * 0.5), angle_deg, scale);
  M.at<double>(0, 2) += tx - (W_in - out_w) * 0.5;
  M.at<double>(1, 2) += ty - (H_in - out_h) * 0.5;

  // Border color matches YOLO's neutral 114-grey at the depth of
  // whichever cv::Mat we received (uint8 mosaic canvas vs float32
  // single-image tensor).
  cv::Scalar border = (in_img.depth() == CV_32F)
      ? cv::Scalar(114.0 / 255.0, 114.0 / 255.0, 114.0 / 255.0)
      : cv::Scalar(114, 114, 114);
  cv::Mat warped;
  cv::warpAffine(in_img, warped, M, {out_w, out_h}, cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, border);

  // Bbox transform.
  std::vector<float> kept;
  if (!in_lbls.empty()) {
    const double m00 = M.at<double>(0, 0), m01 = M.at<double>(0, 1), m02 = M.at<double>(0, 2);
    const double m10 = M.at<double>(1, 0), m11 = M.at<double>(1, 1), m12 = M.at<double>(1, 2);
    auto warp = [&](double x, double y, double& xo, double& yo) {
      xo = m00 * x + m01 * y + m02;
      yo = m10 * x + m11 * y + m12;
    };
    kept.reserve(in_lbls.size());
    for (std::size_t k = 0; k + 5 <= in_lbls.size(); k += 5) {
      const float cls = in_lbls[k];
      const float cx  = in_lbls[k + 1], cy = in_lbls[k + 2];
      const float w   = in_lbls[k + 3], h  = in_lbls[k + 4];
      const double x1 = cx - w * 0.5, y1 = cy - h * 0.5;
      const double x2 = cx + w * 0.5, y2 = cy + h * 0.5;
      double cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4;
      warp(x1, y1, cx1, cy1);
      warp(x2, y1, cx2, cy2);
      warp(x2, y2, cx3, cy3);
      warp(x1, y2, cx4, cy4);
      double nx1 = std::min(std::min(cx1, cx2), std::min(cx3, cx4));
      double ny1 = std::min(std::min(cy1, cy2), std::min(cy3, cy4));
      double nx2 = std::max(std::max(cx1, cx2), std::max(cx3, cx4));
      double ny2 = std::max(std::max(cy1, cy2), std::max(cy3, cy4));
      nx1 = std::clamp(nx1, 0.0, (double)out_w);
      ny1 = std::clamp(ny1, 0.0, (double)out_h);
      nx2 = std::clamp(nx2, 0.0, (double)out_w);
      ny2 = std::clamp(ny2, 0.0, (double)out_h);
      const double nw = nx2 - nx1, nh = ny2 - ny1;
      if (nw < 2.0 || nh < 2.0) continue;
      const double orig_area = (double)w * (double)h;
      if (orig_area > 0.0 && (nw * nh) / orig_area < 0.10) continue;
      kept.push_back(cls);
      kept.push_back((float)((nx1 + nx2) * 0.5));
      kept.push_back((float)((ny1 + ny2) * 0.5));
      kept.push_back((float)nw);
      kept.push_back((float)nh);
    }
  }
  return {warped, std::move(kept)};
}

// Mixup (#54D): blend two examples by α ~ Beta(8, 8); concatenate
// their bbox lists. Both examples must share spatial dims (true
// when they come from the same imgsz).
static YoloExample apply_mixup(const YoloExample& a, const YoloExample& b,
                                std::mt19937& rng) {
  // Beta(8, 8) ≈ symmetric around 0.5; approximate via mean of 8
  // independent Uniform(0,1) variates ratio (close enough for aug).
  std::uniform_real_distribution<float> u(0.f, 1.f);
  float r1 = 0.f, r2 = 0.f;
  for (int i = 0; i < 8; ++i) { r1 += u(rng); r2 += u(rng); }
  float lam = r1 / (r1 + r2 + 1e-8f);

  YoloExample out;
  out.img = a.img * lam + b.img * (1.f - lam);
  out.orig_w = a.orig_w; out.orig_h = a.orig_h; out.gain = a.gain;
  if (a.targets.size(0) == 0)      out.targets = b.targets.clone();
  else if (b.targets.size(0) == 0) out.targets = a.targets.clone();
  else                              out.targets = torch::cat({a.targets, b.targets}, /*dim=*/0);
  return out;
}

YoloDataset::Batch YoloDataset::sample_batch_from_anchors(
    const std::vector<std::size_t>& anchors, std::mt19937& aux_rng) const {
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  std::vector<YoloExample> exs;
  exs.reserve(anchors.size());
  std::vector<torch::Tensor> imgs;
  std::vector<torch::Tensor> tgts_with_b;
  for (std::size_t i = 0; i < anchors.size(); ++i) {
    YoloExample ex;
    const std::size_t anchor = anchors[i];
    if (aug_.augment && aug_.mosaic_p > 0.f && u01(aux_rng) < aug_.mosaic_p) {
      // Anchor-driven mosaic: tile 0 from the provided index (epoch
      // shuffle guarantees uniqueness), tiles 1-3 still random.
      ex = build_mosaic4(*this, (std::size_t)imgsz_, aug_, aux_rng,
                          (std::int64_t)anchor);
    } else {
      ex = get(anchor, /*aug_seed=*/((uint64_t)aux_rng()) << 32 | i);
    }
    if (aug_.augment && aug_.mixup_p > 0.f && u01(aux_rng) < aug_.mixup_p) {
      // MixUp partner is random (not from the shuffled anchors) —
      // matches Ultralytics' behaviour.
      std::uniform_int_distribution<std::size_t> u(0, img_paths_.size() - 1);
      auto other = get(u(aux_rng),
                       /*aug_seed=*/((uint64_t)aux_rng()) << 32 | (i + 0x55));
      ex = apply_mixup(ex, other, aux_rng);
    }
    auto t = ex.targets;
    imgs.push_back(ex.img);
    if (t.size(0) > 0) {
      auto bcol = torch::full({t.size(0), 1}, (double)i, torch::kFloat32);
      tgts_with_b.push_back(torch::cat({bcol, t}, /*dim=*/1));
    }
    exs.push_back(std::move(ex));
  }
  Batch out;
  out.imgs = torch::stack(imgs, /*dim=*/0);
  out.targets = tgts_with_b.empty()
                  ? torch::zeros({0, 6}, torch::kFloat32)
                  : torch::cat(tgts_with_b, /*dim=*/0);
  out.examples = std::move(exs);
  return out;
}

YoloDataset::Batch YoloDataset::sample_batch(std::size_t bsz,
                                             std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, img_paths_.size() - 1);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  std::vector<YoloExample> exs;
  exs.reserve(bsz);
  std::vector<torch::Tensor> imgs;
  std::vector<torch::Tensor> tgts_with_b;

  for (size_t i = 0; i < bsz; ++i) {
    YoloExample ex;
    if (aug_.augment && aug_.mosaic_p > 0.f && u01(rng) < aug_.mosaic_p) {
      // Mosaic + RandomPerspective is combined: build_mosaic4 stitches
      // the 2s canvas and then calls random_perspective_mat to warp
      // and crop to (imgsz, imgsz) in one cv::warpAffine. This matches
      // Ultralytics' pipeline order and avoids the dead-pixel issue
      // you get when stacking perspective on top of an already-cropped
      // mosaic. Non-mosaic samples come from get() pre-letterboxed to
      // imgsz×imgsz and skip the perspective stage entirely (warping
      // them would only create dead pixels at the borders).
      ex = build_mosaic4(*this, (std::size_t)imgsz_, aug_, rng);
    } else {
      ex = get(u(rng), /*aug_seed=*/((uint64_t)rng()) << 32 | i);
    }
    if (aug_.augment && aug_.mixup_p > 0.f && u01(rng) < aug_.mixup_p) {
      auto other = get(u(rng), /*aug_seed=*/((uint64_t)rng()) << 32 | (i + 0x55));
      ex = apply_mixup(ex, other, rng);
    }
    auto t = ex.targets;
    imgs.push_back(ex.img);
    if (t.size(0) > 0) {
      auto bcol = torch::full({t.size(0), 1}, (double)i, torch::kFloat32);
      tgts_with_b.push_back(torch::cat({bcol, t}, /*dim=*/1));
    }
    exs.push_back(std::move(ex));
  }
  Batch out;
  out.imgs = torch::stack(imgs, /*dim=*/0);
  out.targets = tgts_with_b.empty()
                  ? torch::zeros({0, 6}, torch::kFloat32)
                  : torch::cat(tgts_with_b, /*dim=*/0);
  out.examples = std::move(exs);
  return out;
}

}  // namespace yolocpp::datasets
