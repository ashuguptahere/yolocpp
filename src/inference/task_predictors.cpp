#include "yolocpp/inference/task_predictors.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace yolocpp::inference {

static torch::Device pick_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}

// Ultralytics' classify path applies *no* extra normalization on top of
// the standard 0..1 range — it relies on whatever stats the model was
// trained with (which for v8-cls is just /255). Kept as a hook in case
// users need real ImageNet stats later.
static torch::Tensor imagenet_normalize(torch::Tensor img) { return img; }

// ─── ClassifyPredictor ─────────────────────────────────────────────────────
ClassifyPredictor::ClassifyPredictor(const std::string& weights, int imgsz,
                                     std::string device, int nc,
                                     models::YoloV8Scale scale)
    : model_(scale, nc),
      device_(pick_device(std::move(device))),
      imgsz_(imgsz) {
  auto sd = serialization::load_state_dict(weights);
  int copied = model_->load_from_state_dict(sd.entries);
  std::cout << "[classify] loaded " << copied << " tensors from " << weights << "\n";
  model_->to(device_);
  model_->eval();
}

ClassifyResult ClassifyPredictor::predict(const cv::Mat& bgr, int top_k) const {
  // Ultralytics classify preprocessing: resize-shortest-side then center-crop
  // to imgsz×imgsz, BGR→RGB, /255, no further normalization.
  cv::Mat resized;
  int short_side = std::min(bgr.rows, bgr.cols);
  double scale = (double)imgsz_ / short_side;
  cv::resize(bgr, resized, {}, scale, scale, cv::INTER_LINEAR);
  int x0 = (resized.cols - imgsz_) / 2;
  int y0 = (resized.rows - imgsz_) / 2;
  cv::Mat cropped = resized(cv::Rect(x0, y0, imgsz_, imgsz_)).clone();

  auto x = image_to_tensor(cropped).unsqueeze(0).to(device_);
  x = imagenet_normalize(x);

  torch::Tensor logits;
  {
    torch::NoGradGuard ng;
    logits = const_cast<models::YoloV8Classify&>(model_)->forward(x);
  }
  auto probs = logits.softmax(/*dim=*/1).cpu();
  auto k     = std::min(top_k, (int)probs.size(1));
  auto topk  = probs.topk(k, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
  auto vals  = std::get<0>(topk).accessor<float, 2>();
  auto idx   = std::get<1>(topk).accessor<int64_t, 2>();
  ClassifyResult r;
  for (int i = 0; i < k; ++i)
    r.topk.emplace_back((int)idx[0][i], vals[0][i]);
  return r;
}

const std::vector<std::string>& imagenet_names() {
  static const std::vector<std::string> n = {};
  // Loading 1000 names inline is ~30 KB; for now we display class ids.
  return n;
}

// ─── SegmentPredictor ─────────────────────────────────────────────────────
SegmentPredictor::SegmentPredictor(const std::string& weights, int imgsz,
                                   std::string device, int nc,
                                   models::YoloV8Scale scale)
    : model_(scale, nc),
      device_(pick_device(std::move(device))),
      imgsz_(imgsz) {
  auto sd = serialization::load_state_dict(weights);
  int copied = model_->load_from_state_dict(sd.entries);
  std::cout << "[segment] loaded " << copied << " tensors from " << weights << "\n";
  model_->to(device_);
  model_->eval();
}

std::vector<SegInstance> SegmentPredictor::predict(const cv::Mat& bgr,
                                                    NMSConfig nmscfg) const {
  auto lb = letterbox(bgr, imgsz_);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(device_);

  torch::Tensor pred, coefs, protos;
  {
    torch::NoGradGuard ng;
    auto out = const_cast<models::YoloV8Segment&>(model_)->forward_eval(x);
    pred   = std::get<0>(out);
    coefs  = std::get<1>(out);
    protos = std::get<2>(out);
  }

  // Run normal NMS on detection output (pred shape [N=1, 4+nc, A]).
  // We need the surviving anchor indices to also pull from coefs, so
  // re-implement NMS lightly here against the per-image case.
  auto p = pred.transpose(1, 2).contiguous().to(torch::kCPU)[0];   // [A, 4+nc]
  auto box  = p.slice(1, 0, 4);
  auto cls  = p.slice(1, 4, p.size(1));
  auto best = cls.max(1);
  auto conf = std::get<0>(best);
  auto cidx = std::get<1>(best);

  auto mask = conf > nmscfg.conf_thresh;
  auto idx  = torch::nonzero(mask).flatten();
  if (idx.numel() == 0) return {};
  box   = box.index_select(0, idx);
  conf  = conf.index_select(0, idx);
  cidx  = cidx.index_select(0, idx);
  auto sel_coefs = coefs.transpose(1, 2)[0].to(torch::kCPU)
                       .index_select(0, idx);  // [k, nm]

  if (idx.numel() > nmscfg.max_nms) {
    auto topk = conf.topk(nmscfg.max_nms);
    auto ti = std::get<1>(topk);
    box   = box.index_select(0, ti);
    conf  = conf.index_select(0, ti);
    cidx  = cidx.index_select(0, ti);
    sel_coefs = sel_coefs.index_select(0, ti);
  }

  // Class-aware NMS via offset trick.
  auto offset = cidx.to(box.dtype()) * 7680.f;
  auto box_off = box.clone();
  for (int c = 0; c < 4; ++c) box_off.select(1, c).add_(offset);
  // CPU NMS (re-using nms_cpu logic via at::ops would need linkage; we just
  // re-implement here to keep the predictor self-contained).
  auto sorted = conf.argsort(-1, /*descending=*/true);
  auto a_box  = box_off.accessor<float, 2>();
  std::vector<bool> sup(box.size(0), false);
  std::vector<int64_t> keep;
  auto a_idx = sorted.accessor<int64_t, 1>();
  for (int64_t i = 0; i < sorted.size(0); ++i) {
    if (sup[i]) continue;
    auto ii = a_idx[i];
    keep.push_back(ii);
    float xi1 = a_box[ii][0], yi1 = a_box[ii][1],
          xi2 = a_box[ii][2], yi2 = a_box[ii][3];
    float ai = (xi2 - xi1) * (yi2 - yi1);
    for (int64_t j = i + 1; j < sorted.size(0); ++j) {
      if (sup[j]) continue;
      auto jj = a_idx[j];
      float xj1 = a_box[jj][0], yj1 = a_box[jj][1],
            xj2 = a_box[jj][2], yj2 = a_box[jj][3];
      float xx1 = std::max(xi1, xj1), yy1 = std::max(yi1, yj1);
      float xx2 = std::min(xi2, xj2), yy2 = std::min(yi2, yj2);
      float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
      float inter = w * h, aj = (xj2 - xj1) * (yj2 - yj1);
      float iou = inter / (ai + aj - inter + 1e-7f);
      if (iou > nmscfg.iou_thresh) sup[j] = true;
    }
  }
  if ((int)keep.size() > nmscfg.max_det) keep.resize(nmscfg.max_det);

  auto keep_t = torch::from_blob(keep.data(), {(int64_t)keep.size()},
                                  torch::kLong).clone();
  box       = box.index_select(0, keep_t);
  conf      = conf.index_select(0, keep_t);
  cidx      = cidx.index_select(0, keep_t);
  sel_coefs = sel_coefs.index_select(0, keep_t);

  // Compute masks: matmul(coefs [k, nm], protos [N=1, nm, h, w].view) → [k, h, w]
  auto p_cpu = protos.to(torch::kCPU)[0];           // [nm, h_p, w_p]
  auto h_p = p_cpu.size(1), w_p = p_cpu.size(2);
  auto p_flat = p_cpu.reshape({p_cpu.size(0), h_p * w_p});
  auto masks  = sel_coefs.matmul(p_flat).reshape({-1, h_p, w_p}).sigmoid();

  // Convert boxes to original image coords.
  auto box_xyxy = box.clone();
  scale_boxes(box_xyxy, lb);

  // Build instances.
  std::vector<SegInstance> out;
  out.reserve(keep.size());
  auto a_b = box_xyxy.accessor<float, 2>();
  auto a_c = conf.accessor<float, 1>();
  auto a_l = cidx.accessor<int64_t, 1>();

  for (int64_t i = 0; i < (int64_t)keep.size(); ++i) {
    SegInstance inst;
    inst.box.x1   = a_b[i][0];
    inst.box.y1   = a_b[i][1];
    inst.box.x2   = a_b[i][2];
    inst.box.y2   = a_b[i][3];
    inst.box.conf = a_c[i];
    inst.box.cls  = (int)a_l[i];

    // Mask: threshold at 0.5 in proto-resolution, then resize to letterbox
    // image (imgsz × imgsz), then crop to the unpadded region, then resize
    // to original image size.
    auto m = masks[i].gt(0.5).to(torch::kU8) * 255;
    cv::Mat mat(h_p, w_p, CV_8UC1, m.data_ptr<uint8_t>());
    cv::Mat lb_mask;
    cv::resize(mat, lb_mask, {imgsz_, imgsz_}, 0, 0, cv::INTER_LINEAR);
    int unpad_w = (int)std::round(lb.orig_w * lb.gain);
    int unpad_h = (int)std::round(lb.orig_h * lb.gain);
    int x0 = (int)std::round(lb.pad_x);
    int y0 = (int)std::round(lb.pad_y);
    cv::Rect roi(x0, y0,
                 std::min(unpad_w, lb_mask.cols - x0),
                 std::min(unpad_h, lb_mask.rows - y0));
    if (roi.width <= 0 || roi.height <= 0) {
      inst.mask = cv::Mat::zeros(lb.orig_h, lb.orig_w, CV_8UC1);
    } else {
      cv::Mat roi_mask = lb_mask(roi);
      cv::Mat full;
      cv::resize(roi_mask, full, {lb.orig_w, lb.orig_h}, 0, 0,
                 cv::INTER_LINEAR);
      inst.mask = full > 127;
    }
    out.push_back(std::move(inst));
  }
  return out;
}

std::vector<SegInstance> SegmentPredictor::predict_to_file(
    const std::string& in_path, const std::string& out_path, NMSConfig conf,
    const std::vector<std::string>& names) const {
  cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
  if (img.empty())
    throw std::runtime_error("could not read " + in_path);
  auto insts = predict(img, conf);
  const auto& nm = names.empty() ? coco_names() : names;
  cv::Mat overlay = img.clone();
  for (const auto& inst : insts) {
    cv::Scalar color((inst.box.cls * 41) % 256,
                     (inst.box.cls * 73) % 256,
                     (inst.box.cls * 11) % 256);
    overlay.setTo(color, inst.mask);
    cv::rectangle(img, {(int)inst.box.x1, (int)inst.box.y1},
                  {(int)inst.box.x2, (int)inst.box.y2}, color, 2);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f",
                  (inst.box.cls < (int)nm.size() ? nm[inst.box.cls].c_str()
                                                  : "?"),
                  inst.box.conf);
    cv::putText(img, buf, {(int)inst.box.x1 + 2, (int)inst.box.y1 + 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  cv::Mat blended;
  cv::addWeighted(img, 0.6, overlay, 0.4, 0, blended);
  if (!cv::imwrite(out_path, blended))
    throw std::runtime_error("could not write " + out_path);
  return insts;
}

// ─── PosePredictor ────────────────────────────────────────────────────────
PosePredictor::PosePredictor(const std::string& weights, int imgsz,
                              std::string device, int num_kpts, int kpt_dim,
                              models::YoloV8Scale scale)
    : model_(scale, /*nc=*/1, num_kpts, kpt_dim),
      device_(pick_device(std::move(device))),
      imgsz_(imgsz),
      num_kpts_(num_kpts), kpt_dim_(kpt_dim) {
  auto sd = serialization::load_state_dict(weights);
  int copied = model_->load_from_state_dict(sd.entries);
  std::cout << "[pose] loaded " << copied << " tensors from " << weights << "\n";
  model_->to(device_);
  model_->eval();
}

std::vector<PoseInstance> PosePredictor::predict(const cv::Mat& bgr,
                                                  NMSConfig nmscfg) const {
  auto lb = letterbox(bgr, imgsz_);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(device_);

  torch::Tensor pred, kpts;
  {
    torch::NoGradGuard ng;
    auto out = const_cast<models::YoloV8Pose&>(model_)->forward_eval(x);
    pred = std::get<0>(out);
    kpts = std::get<1>(out);
  }

  // NMS on pred (single class).
  auto outs = nms(pred, nmscfg);
  auto& det = outs[0];           // [k, 6]
  if (det.size(0) == 0) return {};

  // We need the indices that survived NMS. Re-run a quick reduction to
  // align — the simpler approach is to run NMS in the same "filter+sort"
  // pattern locally so we can index kpts. To stay concise, recompute the
  // confidence mask & sorting on CPU and reapply.
  auto p = pred.transpose(1, 2).contiguous().to(torch::kCPU)[0];
  auto conf = p.slice(1, 4, 4 + 1).squeeze(1);          // single class
  auto mask = conf > nmscfg.conf_thresh;
  auto idx  = torch::nonzero(mask).flatten();
  if (idx.numel() == 0) return {};

  // Sort surviving anchors by confidence desc and take same NMS keep set.
  auto box  = p.slice(1, 0, 4).index_select(0, idx);
  auto cf   = conf.index_select(0, idx);
  auto sorted = cf.argsort(-1, true);
  auto box_s  = box.index_select(0, sorted);
  auto cf_s   = cf.index_select(0, sorted);
  std::vector<bool> sup(box_s.size(0), false);
  std::vector<int64_t> keep;
  auto a = box_s.accessor<float, 2>();
  for (int64_t i = 0; i < box_s.size(0); ++i) {
    if (sup[i]) continue;
    keep.push_back(i);
    float xi1 = a[i][0], yi1 = a[i][1], xi2 = a[i][2], yi2 = a[i][3];
    float ai = (xi2 - xi1) * (yi2 - yi1);
    for (int64_t j = i + 1; j < box_s.size(0); ++j) {
      if (sup[j]) continue;
      float xj1 = a[j][0], yj1 = a[j][1], xj2 = a[j][2], yj2 = a[j][3];
      float xx1 = std::max(xi1, xj1), yy1 = std::max(yi1, yj1);
      float xx2 = std::min(xi2, xj2), yy2 = std::min(yi2, yj2);
      float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
      float inter = w * h, aj = (xj2 - xj1) * (yj2 - yj1);
      if (inter / (ai + aj - inter + 1e-7f) > nmscfg.iou_thresh) sup[j] = true;
    }
  }
  if ((int)keep.size() > nmscfg.max_det) keep.resize(nmscfg.max_det);

  // Map kpts (indexed by original anchor) through idx → sorted → keep.
  auto kpts_cpu = kpts.transpose(1, 2)[0].to(torch::kCPU);  // [A, nk]
  auto kpts_sel = kpts_cpu.index_select(0, idx);
  auto kpts_sorted = kpts_sel.index_select(0, sorted);
  auto keep_t = torch::from_blob(keep.data(), {(int64_t)keep.size()},
                                  torch::kLong).clone();
  auto box_keep   = box_s.index_select(0, keep_t);
  auto cf_keep    = cf_s.index_select(0, keep_t);
  auto kpts_keep  = kpts_sorted.index_select(0, keep_t);

  // Unscale boxes and keypoints.
  scale_boxes(box_keep, lb);
  // For keypoints: subtract pad, divide by gain. kpts_keep is [k, nk] where
  // nk = num_kpts * 3 = (x, y, conf, x, y, conf, ...).
  auto nk = num_kpts_ * kpt_dim_;
  auto k2 = kpts_keep.reshape({-1, num_kpts_, kpt_dim_});  // [k, K, 3]
  k2.select(2, 0).sub_(lb.pad_x).div_(lb.gain);
  k2.select(2, 1).sub_(lb.pad_y).div_(lb.gain);

  std::vector<PoseInstance> out;
  out.reserve(keep.size());
  auto a_b = box_keep.accessor<float, 2>();
  auto a_c = cf_keep.accessor<float, 1>();
  auto a_k = k2.accessor<float, 3>();
  for (int64_t i = 0; i < (int64_t)keep.size(); ++i) {
    PoseInstance pi;
    pi.box.x1 = a_b[i][0]; pi.box.y1 = a_b[i][1];
    pi.box.x2 = a_b[i][2]; pi.box.y2 = a_b[i][3];
    pi.box.conf = a_c[i];
    pi.box.cls  = 0;
    pi.keypoints.resize(num_kpts_);
    for (int j = 0; j < num_kpts_; ++j)
      pi.keypoints[j] = {a_k[i][j][0], a_k[i][j][1], a_k[i][j][2]};
    out.push_back(std::move(pi));
  }
  return out;
}

std::vector<PoseInstance> PosePredictor::predict_to_file(
    const std::string& in_path, const std::string& out_path, NMSConfig conf) const {
  cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
  if (img.empty()) throw std::runtime_error("could not read " + in_path);
  auto insts = predict(img, conf);

  // COCO 17-keypoint skeleton edges.
  static const int edges[][2] = {
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
      {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
      {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6},
  };
  for (const auto& inst : insts) {
    cv::rectangle(img, {(int)inst.box.x1, (int)inst.box.y1},
                  {(int)inst.box.x2, (int)inst.box.y2}, {0, 255, 0}, 2);
    for (auto& kp : inst.keypoints) {
      if (kp[2] > 0.5f)
        cv::circle(img, {(int)kp[0], (int)kp[1]}, 3, {0, 0, 255}, -1);
    }
    for (auto& e : edges) {
      const auto& a = inst.keypoints[e[0]];
      const auto& b = inst.keypoints[e[1]];
      if (a[2] > 0.5f && b[2] > 0.5f)
        cv::line(img, {(int)a[0], (int)a[1]}, {(int)b[0], (int)b[1]},
                 {255, 0, 0}, 2);
    }
  }
  if (!cv::imwrite(out_path, img))
    throw std::runtime_error("could not write " + out_path);
  return insts;
}

// ─── OBBPredictor ─────────────────────────────────────────────────────────
OBBPredictor::OBBPredictor(const std::string& weights, int imgsz,
                            std::string device, int nc,
                            models::YoloV8Scale scale)
    : model_(scale, nc, /*ne=*/1),
      device_(pick_device(std::move(device))),
      imgsz_(imgsz) {
  auto sd = serialization::load_state_dict(weights);
  int copied = model_->load_from_state_dict(sd.entries);
  std::cout << "[obb] loaded " << copied << " tensors from " << weights << "\n";
  model_->to(device_);
  model_->eval();
}

// Probabilistic IoU (covariance-form) approx used by Ultralytics for OBB
// NMS. We use a simpler convex approximation: convert (cx,cy,w,h,θ) to
// rotated polygon via cv2.RotatedRect and compute IoU via cv2 helpers.
static double rotated_iou(float cx1, float cy1, float w1, float h1, float a1,
                          float cx2, float cy2, float w2, float h2, float a2) {
  cv::RotatedRect r1(cv::Point2f(cx1, cy1), cv::Size2f(w1, h1),
                     a1 * 180.f / (float)M_PI);
  cv::RotatedRect r2(cv::Point2f(cx2, cy2), cv::Size2f(w2, h2),
                     a2 * 180.f / (float)M_PI);
  std::vector<cv::Point2f> inter;
  int t = cv::rotatedRectangleIntersection(r1, r2, inter);
  if (t == cv::INTERSECT_NONE || inter.size() < 3) return 0.0;
  double inter_a = cv::contourArea(inter);
  double a       = w1 * h1 + w2 * h2 - inter_a;
  return a > 0 ? inter_a / a : 0.0;
}

std::vector<OBBInstance> OBBPredictor::predict(const cv::Mat& bgr,
                                                NMSConfig nmscfg) const {
  auto lb = letterbox(bgr, imgsz_);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(device_);

  torch::Tensor pred, angle;
  {
    torch::NoGradGuard ng;
    auto out = const_cast<models::YoloV8OBB&>(model_)->forward_eval(x);
    pred  = std::get<0>(out);
    angle = std::get<1>(out);
  }

  // pred: [1, 4 + nc, A] (xyxy + cls). Convert xyxy back to (cx, cy, w, h).
  auto p = pred.transpose(1, 2).contiguous().to(torch::kCPU)[0];   // [A, 4+nc]
  auto box  = p.slice(1, 0, 4);
  auto cls  = p.slice(1, 4, p.size(1));
  auto best = cls.max(1);
  auto conf = std::get<0>(best);
  auto cidx = std::get<1>(best);
  auto a_cpu = angle.to(torch::kCPU)[0];                            // [A]

  auto mask = conf > nmscfg.conf_thresh;
  auto idx  = torch::nonzero(mask).flatten();
  if (idx.numel() == 0) return {};

  box = box.index_select(0, idx);
  conf = conf.index_select(0, idx);
  cidx = cidx.index_select(0, idx);
  auto ag  = a_cpu.index_select(0, idx);

  // Sort by confidence and rotated NMS.
  auto sorted = conf.argsort(-1, true);
  box  = box.index_select(0, sorted);
  conf = conf.index_select(0, sorted);
  cidx = cidx.index_select(0, sorted);
  ag   = ag.index_select(0, sorted);

  auto a_b = box.accessor<float, 2>();
  auto a_c = conf.accessor<float, 1>();
  auto a_l = cidx.accessor<int64_t, 1>();
  auto a_a = ag.accessor<float, 1>();

  // Convert to (cx, cy, w, h) in letterbox coords.
  std::vector<std::array<float, 5>> rrects(box.size(0));
  for (int64_t i = 0; i < box.size(0); ++i) {
    float x1 = a_b[i][0], y1 = a_b[i][1], x2 = a_b[i][2], y2 = a_b[i][3];
    rrects[i] = {(x1 + x2) * 0.5f, (y1 + y2) * 0.5f, x2 - x1, y2 - y1, a_a[i]};
  }

  std::vector<bool> sup(rrects.size(), false);
  std::vector<int64_t> keep;
  for (int64_t i = 0; i < (int64_t)rrects.size(); ++i) {
    if (sup[i]) continue;
    keep.push_back(i);
    for (int64_t j = i + 1; j < (int64_t)rrects.size(); ++j) {
      if (sup[j]) continue;
      if (a_l[i] != a_l[j]) continue;  // class-aware
      double iou = rotated_iou(rrects[i][0], rrects[i][1],
                               rrects[i][2], rrects[i][3], rrects[i][4],
                               rrects[j][0], rrects[j][1],
                               rrects[j][2], rrects[j][3], rrects[j][4]);
      if (iou > nmscfg.iou_thresh) sup[j] = true;
    }
  }
  if ((int)keep.size() > nmscfg.max_det) keep.resize(nmscfg.max_det);

  std::vector<OBBInstance> out;
  out.reserve(keep.size());
  for (auto i : keep) {
    OBBInstance o;
    // Unscale (cx, cy) and dimensions back to original image coords.
    o.cx    = (rrects[i][0] - (float)lb.pad_x) / (float)lb.gain;
    o.cy    = (rrects[i][1] - (float)lb.pad_y) / (float)lb.gain;
    o.w     =  rrects[i][2] / (float)lb.gain;
    o.h     =  rrects[i][3] / (float)lb.gain;
    o.angle = rrects[i][4];
    o.conf  = a_c[i];
    o.cls   = (int)a_l[i];
    out.push_back(o);
  }
  return out;
}

std::vector<OBBInstance> OBBPredictor::predict_to_file(
    const std::string& in_path, const std::string& out_path, NMSConfig conf,
    const std::vector<std::string>& names) const {
  cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
  if (img.empty()) throw std::runtime_error("could not read " + in_path);
  auto insts = predict(img, conf);
  const auto& nm = names.empty() ? dota_names() : names;
  for (const auto& o : insts) {
    cv::RotatedRect rr({o.cx, o.cy}, {o.w, o.h}, o.angle * 180.f / (float)M_PI);
    cv::Point2f pts[4];
    rr.points(pts);
    cv::Scalar color((o.cls * 41) % 256, (o.cls * 73) % 256,
                     (o.cls * 11) % 256);
    for (int i = 0; i < 4; ++i)
      cv::line(img, pts[i], pts[(i + 1) % 4], color, 2);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f",
                  (o.cls < (int)nm.size() ? nm[o.cls].c_str() : "?"),
                  o.conf);
    cv::putText(img, buf, {(int)pts[1].x, (int)pts[1].y - 4},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
  }
  if (!cv::imwrite(out_path, img))
    throw std::runtime_error("could not write " + out_path);
  return insts;
}

const std::vector<std::string>& dota_names() {
  static const std::vector<std::string> n = {
      "plane", "ship", "storage tank", "baseball diamond", "tennis court",
      "basketball court", "ground track field", "harbor", "bridge",
      "large vehicle", "small vehicle", "helicopter", "roundabout",
      "soccer ball field", "swimming pool",
  };
  return n;
}

}  // namespace yolocpp::inference
