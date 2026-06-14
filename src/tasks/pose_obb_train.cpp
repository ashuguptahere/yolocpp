#include "yolocpp/tasks/pose_obb_train.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/optim.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
#include <stdexcept>

#include "yolocpp/inference/letterbox.hpp"

namespace fs = std::filesystem;

namespace yolocpp::tasks {

namespace {

torch::Device pick_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}
std::vector<std::string> list_imgs(const fs::path& d) {
  std::vector<std::string> out;
  if (!fs::exists(d)) return out;
  for (auto& e : fs::directory_iterator(d)) {
    if (!e.is_regular_file()) continue;
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      out.push_back(e.path().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}
std::string label_path_for(const std::string& img_path) {
  std::string s = img_path;
  auto pos = s.rfind("/images/");
  if (pos != std::string::npos) s.replace(pos, 8, "/labels/");
  fs::path p(s);
  p.replace_extension(".txt");
  return p.string();
}

}  // anonymous namespace

// ============================================================================
// PoseDataset
// ============================================================================
PoseDataset::PoseDataset(std::string root, std::string split, int imgsz,
                         int num_kpts, int kpt_dim, bool augment)
    : imgsz_(imgsz), nk_(num_kpts), kdim_(kpt_dim), augment_(augment) {
  fs::path img_dir = fs::path(root) / "images" / split;
  paths_ = list_imgs(img_dir);
  for (auto& p : paths_) lbl_paths_.push_back(label_path_for(p));
  if (paths_.empty()) throw std::runtime_error("no images at " + img_dir.string());
}

PoseExample PoseDataset::get(std::size_t idx, uint64_t seed) const {
  PoseExample ex;
  ex.img_path = paths_[idx];
  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("cannot read " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;
  auto lb = inference::letterbox(bgr, imgsz_);
  ex.gain = lb.gain; ex.pad_x = lb.pad_x; ex.pad_y = lb.pad_y;

  std::mt19937 rng(seed ? seed : (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  bool flip = false;
  if (augment_) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < 0.5f;
    if (flip) cv::flip(lb.img, lb.img, 1);
  }
  ex.img = inference::image_to_tensor(lb.img);

  // Parse "cls cx cy w h k1x k1y v1 ..."
  std::ifstream f(lbl_paths_[idx]);
  std::vector<float> tgt_rows;
  std::vector<float> kpt_rows;
  int n_rows = 0;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int cls;
    float cx, cy, w, h;
    if (!(iss >> cls >> cx >> cy >> w >> h)) continue;
    cx = (float)((cx * ex.orig_w) * lb.gain + lb.pad_x);
    cy = (float)((cy * ex.orig_h) * lb.gain + lb.pad_y);
    w  = (float)((w  * ex.orig_w) * lb.gain);
    h  = (float)((h  * ex.orig_h) * lb.gain);
    if (flip) cx = (float)imgsz_ - 1.f - cx;
    tgt_rows.push_back((float)cls);
    tgt_rows.push_back(cx); tgt_rows.push_back(cy);
    tgt_rows.push_back(w);  tgt_rows.push_back(h);
    // Keypoints
    for (int k = 0; k < nk_; ++k) {
      float kx, ky, v;
      if (!(iss >> kx >> ky >> v)) { kx = ky = v = 0; }
      kx = (float)((kx * ex.orig_w) * lb.gain + lb.pad_x);
      ky = (float)((ky * ex.orig_h) * lb.gain + lb.pad_y);
      if (flip && v > 0) kx = (float)imgsz_ - 1.f - kx;
      kpt_rows.push_back(kx); kpt_rows.push_back(ky); kpt_rows.push_back(v);
    }
    ++n_rows;
  }
  if (n_rows == 0) {
    ex.targets   = torch::zeros({0, 5}, torch::kFloat32);
    ex.keypoints = torch::zeros({0, nk_, 3}, torch::kFloat32);
    return ex;
  }
  ex.targets = torch::from_blob(tgt_rows.data(), {n_rows, 5},
                                torch::kFloat32).clone();
  ex.keypoints = torch::from_blob(kpt_rows.data(), {n_rows, nk_, 3},
                                  torch::kFloat32).clone();
  return ex;
}

PoseDataset::Batch PoseDataset::sample_batch(std::size_t bsz,
                                              std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, paths_.size() - 1);
  std::vector<torch::Tensor> imgs, tgts, kpts;
  std::vector<PoseExample> exs;
  for (size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), (uint64_t)rng() << 32 | i);
    imgs.push_back(ex.img);
    if (ex.targets.size(0) > 0) {
      auto bcol = torch::full({ex.targets.size(0), 1}, (double)i, torch::kFloat32);
      tgts.push_back(torch::cat({bcol, ex.targets}, 1));
      kpts.push_back(ex.keypoints);
    }
    exs.push_back(std::move(ex));
  }
  Batch b;
  b.imgs    = torch::stack(imgs, 0);
  b.targets = tgts.empty() ? torch::zeros({0, 6}, torch::kFloat32)
                           : torch::cat(tgts, 0);
  b.keypoints = kpts.empty() ? torch::zeros({0, nk_, 3}, torch::kFloat32)
                             : torch::cat(kpts, 0);
  b.examples = std::move(exs);
  return b;
}

// ============================================================================
// train_pose — minimal: keypoint L1 + visibility BCE on the closest anchor
// ============================================================================
template <typename M>
void train_pose_t(M model,
                const PoseDataset& train,
                const PoseDataset* /*val*/,
                PoseTrainConfig cfg) {
  fs::create_directories(cfg.save_dir);
  auto device = pick_device(cfg.device);
  model->to(device);
  model->train();

  std::vector<at::Tensor> params;
  for (auto& p : model->parameters())
    if (p.requires_grad()) params.push_back(p);
  auto opts = torch::optim::SGDOptions(cfg.lr0)
                  .momentum(cfg.momentum)
                  .weight_decay(cfg.weight_decay)
                  .nesterov(true);
  torch::optim::SGD optim(params, opts);

  std::mt19937 rng(0x9E3779B9u);
  size_t n = train.size();
  int    steps = std::max<int>(1, (int)(n / cfg.batch_size));
  int    total = steps * cfg.epochs;
  int    warmup = std::max(50, steps * cfg.warmup_epochs);
  std::cout << "[pose-train] " << n << " imgs, " << steps << " steps/epoch, "
            << "device=" << device << "\n";

  double s0 = model->stride[0];
  int    feat_h = cfg.imgsz / (int)s0;
  int    feat_w = cfg.imgsz / (int)s0;
  int    A_lvl  = feat_h * feat_w;

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto t0 = std::chrono::steady_clock::now();
    double sum_kpt = 0;
    for (int step = 0; step < steps; ++step) {
      int gstep = epoch * steps + step;
      double scale;
      if (gstep < warmup) scale = (double)(gstep + 1) / warmup;
      else {
        double t = (double)(gstep - warmup) /
                   std::max(1.0, (double)(total - warmup));
        scale = cfg.lrf + 0.5 * (1.0 - cfg.lrf) * (1.0 + std::cos(M_PI * t));
      }
      auto& gs = optim.param_groups();
      static_cast<torch::optim::SGDOptions&>(gs[0].options()).lr(cfg.lr0 * scale);

      auto b = train.sample_batch(cfg.batch_size, rng);
      auto x = b.imgs.to(device);
      // Forward — get decoded preds + keypoints.
      auto out = model->forward_eval(x);
      auto kpts_pred = std::get<1>(out);   // [B, K*3, A], in pixel coords

      if (b.targets.size(0) == 0) continue;

      // Pick one anchor per target (closest cell at smallest stride).
      auto t_cpu = b.targets.detach();
      auto a = t_cpu.accessor<float, 2>();
      std::vector<int64_t> anc_idxs, batch_idxs;
      std::vector<int> valid;
      for (int i = 0; i < (int)t_cpu.size(0); ++i) {
        int bi = (int)a[i][0];
        float cx = a[i][2], cy = a[i][3];
        int xi = std::clamp((int)std::floor(cx / s0), 0, feat_w - 1);
        int yi = std::clamp((int)std::floor(cy / s0), 0, feat_h - 1);
        anc_idxs.push_back(yi * feat_w + xi);
        batch_idxs.push_back(bi);
        valid.push_back(i);
      }
      auto anc_t = torch::tensor(anc_idxs,
                                 torch::TensorOptions().device(device).dtype(torch::kLong));
      auto bat_t = torch::tensor(batch_idxs,
                                 torch::TensorOptions().device(device).dtype(torch::kLong));
      auto kpts_lvl0 = kpts_pred.slice(2, 0, A_lvl);                  // [B, K*3, A_lvl]
      auto pred_pos  = kpts_lvl0.index({bat_t, torch::indexing::Slice(),
                                          anc_t}).contiguous();        // [P, K*3]
      auto P = pred_pos.size(0);
      auto K = train.num_kpts();
      pred_pos = pred_pos.reshape({P, K, 3});

      // GT keypoints for the valid rows.
      auto gt_kpts = b.keypoints.to(device);   // [P, K, 3]
      // Loss: L1 on (x, y) where v > 0; BCE on v.
      auto vis = (gt_kpts.select(2, 2) > 0.5f).to(torch::kFloat32);
      auto xy_diff = (pred_pos.slice(2, 0, 2) - gt_kpts.slice(2, 0, 2)).abs();
      auto kpt_loss = (xy_diff.sum(-1) * vis).sum() /
                       (vis.sum().clamp_min(1.0) * 2.0);
      auto vis_pred = pred_pos.select(2, 2);  // already sigmoid'd in forward
      auto vis_loss = torch::nn::functional::binary_cross_entropy(
          vis_pred.clamp(1e-6, 1 - 1e-6), vis);

      auto loss = kpt_loss * cfg.kpt_gain + vis_loss * cfg.kobj_gain;
      optim.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model->parameters(), 10.0);
      optim.step();

      sum_kpt += kpt_loss.template item<double>();
      if (gstep % cfg.log_every == 0)
        std::cout << "[pose-train] e=" << epoch << " s=" << step
                  << " kpt=" << kpt_loss.template item<double>()
                  << " vis=" << vis_loss.template item<double>() << "\n";
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[pose-train] epoch " << epoch
              << " avg_kpt=" << (sum_kpt / steps)
              << " in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";
  }
  auto ckpt = fs::path(cfg.save_dir) / "last.pt";
  torch::save(model, ckpt.string());
  std::cout << "[pose-train] saved → " << ckpt << "\n";
}

template <typename M>
PoseValResult validate_pose_t(M& model,
                             const PoseDataset& dataset,
                             torch::Device device) {
  model->to(device); model->eval();
  // COCO 17-keypoint OKS sigmas (per-keypoint falloff). Previously the OKS used
  // a constant sigma=1 and a constant 50px area scale → scale/keypoint-blind;
  // and the metric was recall (every GT matched by ANY prediction), not AP.
  static const std::vector<float> kSigma = {
      0.026f,0.025f,0.025f,0.035f,0.035f,0.079f,0.079f,0.072f,0.072f,
      0.062f,0.062f,0.107f,0.107f,0.087f,0.087f,0.089f,0.089f};
  const int K = dataset.num_kpts();
  std::vector<std::pair<float,int>> dets;   // (conf, tp) — single "person" class
  int n_pred = 0, n_gt = 0;
  for (size_t i = 0; i < dataset.size(); ++i) {
    auto ex = dataset.get(i, /*seed=*/i + 1);
    auto x  = ex.img.unsqueeze(0).to(device);
    torch::Tensor pred, kpts;
    {
      torch::NoGradGuard ng;
      auto o = model->forward_eval(x);
      pred = std::get<0>(o);
      kpts = std::get<1>(o);
    }
    auto p    = pred.transpose(1, 2)[0].to(torch::kCPU);
    auto conf = std::get<0>(p.slice(1, 4, p.size(1)).max(1));
    auto idx  = torch::nonzero(conf.gt(0.05f)).flatten();   // low thr for PR curve
    const int G = (int)ex.targets.size(0);
    n_gt += G;
    if (idx.numel() == 0 || G == 0) continue;

    auto k2 = kpts.transpose(1, 2)[0].to(torch::kCPU)
                  .index_select(0, idx).reshape({-1, K, 3}).contiguous();   // [k,K,3]
    auto sel_conf = conf.index_select(0, idx).contiguous();
    const int k = (int)idx.numel();
    n_pred += k;

    auto gt_kp  = ex.keypoints.to(torch::kCPU).contiguous();   // [G,K,3]
    auto tgt_cpu = ex.targets.to(torch::kCPU).contiguous();    // [G,5] cls,cx,cy,w,h
    auto kacc  = k2.accessor<float, 3>();
    auto gacc  = gt_kp.accessor<float, 3>();
    auto tacc  = tgt_cpu.accessor<float, 2>();

    // OKS[k][G] = mean over visible GT keypoints of exp(-d²/(2·area·σ²)).
    std::vector<std::vector<float>> oks(k, std::vector<float>(G, 0.0f));
    for (int g = 0; g < G; ++g) {
      double area = std::max(1.0, (double)tacc[g][3] * (double)tacc[g][4]);  // bbox w*h
      for (int j = 0; j < k; ++j) {
        double acc = 0.0; int vis = 0;
        for (int kp = 0; kp < K; ++kp) {
          if (gacc[g][kp][2] <= 0.5f) continue;  // GT keypoint not visible
          ++vis;
          double dx = (double)kacc[j][kp][0] - (double)gacc[g][kp][0];
          double dy = (double)kacc[j][kp][1] - (double)gacc[g][kp][1];
          double s  = (kp < (int)kSigma.size()) ? kSigma[kp] : 0.05f;
          acc += std::exp(-(dx*dx + dy*dy) / (2.0 * area * s * s + 1e-9));
        }
        oks[j][g] = vis ? (float)(acc / vis) : 0.0f;
      }
    }
    // Greedy match predictions (conf desc) to best unmatched GT, OKS > 0.5.
    auto order = std::get<1>(sel_conf.sort(/*dim=*/0, /*descending=*/true)).contiguous();
    auto oa  = order.accessor<int64_t, 1>();
    auto cfa = sel_conf.accessor<float, 1>();
    std::vector<bool> used(G, false);
    for (int oi = 0; oi < k; ++oi) {
      int j = (int)oa[oi];
      float best = 0.0f; int bg = -1;
      for (int g = 0; g < G; ++g)
        if (!used[g] && oks[j][g] > best) { best = oks[j][g]; bg = g; }
    int tp = (best > 0.5f && bg >= 0) ? 1 : 0;
      if (tp) used[bg] = true;
      dets.push_back({cfa[j], tp});
    }
  }
  // AP@OKS0.5 (single person class) — area under the recall→precision curve.
  std::sort(dets.begin(), dets.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
              return a.first > b.first;
            });
  double tp = 0, fp = 0, ap = 0, prev_r = 0;
  for (auto& d : dets) {
    if (d.second) tp += 1; else fp += 1;
    double recall = n_gt ? tp / n_gt : 0.0;
    double precision = tp / (tp + fp);
    ap += (recall - prev_r) * precision;
    prev_r = recall;
  }
  PoseValResult r;
  r.n_pred     = n_pred;
  r.n_gt       = n_gt;
  r.n_matched  = (int)tp;
  r.oks_map_50 = n_gt ? ap : 0.0;
  return r;
}

// ============================================================================
// OBBDataset
// ============================================================================
OBBDataset::OBBDataset(std::string root, std::string split, int imgsz,
                       std::vector<std::string> names, bool augment)
    : names_(std::move(names)), imgsz_(imgsz), augment_(augment) {
  fs::path img_dir = fs::path(root) / "images" / split;
  paths_ = list_imgs(img_dir);
  for (auto& p : paths_) lbl_paths_.push_back(label_path_for(p));
  if (paths_.empty()) throw std::runtime_error("no images at " + img_dir.string());
}

OBBLabelExample OBBDataset::get(std::size_t idx, uint64_t seed) const {
  OBBLabelExample ex;
  ex.img_path = paths_[idx];
  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("cannot read " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;
  auto lb = inference::letterbox(bgr, imgsz_);
  ex.gain = lb.gain; ex.pad_x = lb.pad_x; ex.pad_y = lb.pad_y;

  std::mt19937 rng(seed ? seed : (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  bool flip = false;
  if (augment_) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < 0.5f;
    if (flip) cv::flip(lb.img, lb.img, 1);
  }
  ex.img = inference::image_to_tensor(lb.img);

  std::ifstream f(lbl_paths_[idx]);
  std::string line;
  std::vector<float> rows;
  int n_rows = 0;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    int cls;
    if (!(iss >> cls)) continue;
    std::vector<float> v;
    float t;
    while (iss >> t) v.push_back(t);
    float cx, cy, w, h, ang;
    if (v.size() >= 8) {
      // DOTA / YOLO-OBB: 8-point normalized polygon (x1 y1 … x4 y4) → fit a
      // rotated box with cv::minAreaRect. (The old parser read these corners as
      // cx,cy,w,h,angle → garbage GT, so obb train + val were meaningless.)
      std::vector<cv::Point2f> pts(4);
      for (int kk = 0; kk < 4; ++kk) {
        float px = (float)(v[2 * kk]     * ex.orig_w * lb.gain + lb.pad_x);
        float py = (float)(v[2 * kk + 1] * ex.orig_h * lb.gain + lb.pad_y);
        if (flip) px = (float)imgsz_ - 1.f - px;
        pts[(std::size_t)kk] = {px, py};
      }
      cv::RotatedRect rr = cv::minAreaRect(pts);
      cx = rr.center.x; cy = rr.center.y;
      w = rr.size.width; h = rr.size.height;
      ang = rr.angle * (float)M_PI / 180.f;  // radians
    } else if (v.size() >= 4) {
      // Legacy normalized cx,cy,w,h[,angle].
      cx  = (float)(v[0] * ex.orig_w * lb.gain + lb.pad_x);
      cy  = (float)(v[1] * ex.orig_h * lb.gain + lb.pad_y);
      w   = (float)(v[2] * ex.orig_w * lb.gain);
      h   = (float)(v[3] * ex.orig_h * lb.gain);
      ang = (v.size() >= 5) ? v[4] : 0.f;
      if (flip) { cx = (float)imgsz_ - 1.f - cx; ang = -ang; }
    } else {
      continue;
    }
    rows.push_back((float)cls);
    rows.push_back(cx); rows.push_back(cy);
    rows.push_back(w);  rows.push_back(h);
    rows.push_back(ang);
    ++n_rows;
  }
  ex.targets = (n_rows == 0)
                   ? torch::zeros({0, 6}, torch::kFloat32)
                   : torch::from_blob(rows.data(), {n_rows, 6},
                                      torch::kFloat32).clone();
  return ex;
}

OBBDataset::Batch OBBDataset::sample_batch(std::size_t bsz,
                                            std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, paths_.size() - 1);
  std::vector<torch::Tensor> imgs, tgts;
  std::vector<OBBLabelExample> exs;
  for (size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), (uint64_t)rng() << 32 | i);
    imgs.push_back(ex.img);
    if (ex.targets.size(0) > 0) {
      auto bcol = torch::full({ex.targets.size(0), 1}, (double)i, torch::kFloat32);
      tgts.push_back(torch::cat({bcol, ex.targets}, 1));
    }
    exs.push_back(std::move(ex));
  }
  Batch b;
  b.imgs    = torch::stack(imgs, 0);
  b.targets = tgts.empty() ? torch::zeros({0, 7}, torch::kFloat32)
                           : torch::cat(tgts, 0);
  b.examples = std::move(exs);
  return b;
}

// ============================================================================
// train_obb — minimal: angle prediction loss on closest anchor
// ============================================================================
template <typename M>
void train_obb_t(M model,
               const OBBDataset& train,
               const OBBDataset* /*val*/,
               OBBTrainConfig cfg) {
  fs::create_directories(cfg.save_dir);
  auto device = pick_device(cfg.device);
  model->to(device);
  model->train();

  std::vector<at::Tensor> params;
  for (auto& p : model->parameters())
    if (p.requires_grad()) params.push_back(p);
  auto opts = torch::optim::SGDOptions(cfg.lr0)
                  .momentum(cfg.momentum)
                  .weight_decay(cfg.weight_decay)
                  .nesterov(true);
  torch::optim::SGD optim(params, opts);

  std::mt19937 rng(0x9E3779B9u);
  size_t n = train.size();
  int    steps = std::max<int>(1, (int)(n / cfg.batch_size));
  int    total = steps * cfg.epochs;
  int    warmup = std::max(50, steps * cfg.warmup_epochs);

  double s0 = model->stride[0];
  int    feat_h = cfg.imgsz / (int)s0;
  int    feat_w = cfg.imgsz / (int)s0;
  int    A_lvl  = feat_h * feat_w;
  std::cout << "[obb-train] " << n << " imgs, " << steps
            << " steps/epoch, device=" << device << "\n";

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto t0 = std::chrono::steady_clock::now();
    double sum_a = 0;
    for (int step = 0; step < steps; ++step) {
      int gstep = epoch * steps + step;
      double scale;
      if (gstep < warmup) scale = (double)(gstep + 1) / warmup;
      else {
        double t = (double)(gstep - warmup) /
                   std::max(1.0, (double)(total - warmup));
        scale = cfg.lrf + 0.5 * (1.0 - cfg.lrf) * (1.0 + std::cos(M_PI * t));
      }
      auto& gs = optim.param_groups();
      static_cast<torch::optim::SGDOptions&>(gs[0].options()).lr(cfg.lr0 * scale);

      auto b = train.sample_batch(cfg.batch_size, rng);
      auto x = b.imgs.to(device);
      auto out = model->forward_eval(x);
      auto angle = std::get<1>(out);  // [B, A] — already in radians

      if (b.targets.size(0) == 0) continue;
      auto a = b.targets.accessor<float, 2>();
      std::vector<int64_t> anc_idxs, batch_idxs;
      std::vector<float>   gt_ang;
      for (int i = 0; i < (int)b.targets.size(0); ++i) {
        int bi = (int)a[i][0];
        float cx = a[i][2], cy = a[i][3], ang_v = a[i][6];
        int xi = std::clamp((int)std::floor(cx / s0), 0, feat_w - 1);
        int yi = std::clamp((int)std::floor(cy / s0), 0, feat_h - 1);
        anc_idxs.push_back(yi * feat_w + xi);
        batch_idxs.push_back(bi);
        gt_ang.push_back(ang_v);
      }
      auto anc_t = torch::tensor(anc_idxs,
                                 torch::TensorOptions().device(device).dtype(torch::kLong));
      auto bat_t = torch::tensor(batch_idxs,
                                 torch::TensorOptions().device(device).dtype(torch::kLong));
      auto gt_t  = torch::tensor(gt_ang, torch::kFloat32).to(device);

      auto angle_lvl0 = angle.slice(1, 0, A_lvl);                        // [B, A_lvl]
      auto pred_pos = angle_lvl0.index({bat_t, anc_t});                  // [P]
      // Use 1 - cos(diff) — periodic-friendly angular distance.
      auto loss = (1.0 - (pred_pos - gt_t).cos()).mean() * cfg.angle_gain;
      optim.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model->parameters(), 10.0);
      optim.step();

      sum_a += loss.template item<double>();
      if (gstep % cfg.log_every == 0)
        std::cout << "[obb-train] e=" << epoch << " s=" << step
                  << " angle=" << loss.template item<double>() << "\n";
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[obb-train] epoch " << epoch
              << " avg_angle=" << (sum_a / steps)
              << " in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";
  }
  auto ckpt = fs::path(cfg.save_dir) / "last.pt";
  torch::save(model, ckpt.string());
  std::cout << "[obb-train] saved → " << ckpt << "\n";
}

template <typename M>
OBBValResult validate_obb_t(M& model,
                           const OBBDataset& dataset,
                           torch::Device device) {
  model->to(device); model->eval();
  // Real per-class rotated AP@0.5 (was recall: each GT matched by ANY
  // prediction, no confidence sorting, no FP penalty). Greedy-match
  // confidence-sorted predictions to same-class GT by rotated IoU > 0.5.
  std::map<int, std::vector<std::pair<float,int>>> det_by_class;  // cls → (conf, tp)
  std::map<int, int> gt_count;
  int n_pred = 0, n_gt = 0, n_matched = 0;
  for (size_t i = 0; i < dataset.size(); ++i) {
    auto ex = dataset.get(i, /*seed=*/i + 1);
    auto x  = ex.img.unsqueeze(0).to(device);
    torch::Tensor pred, ang;
    {
      torch::NoGradGuard ng;
      auto o = model->forward_eval(x);
      pred = std::get<0>(o); ang = std::get<1>(o);
    }
    const int G = (int)ex.targets.size(0);
    n_gt += G;
    auto p = pred.transpose(1, 2)[0].to(torch::kCPU);
    auto cf = p.slice(1, 4, p.size(1)).max(1);
    auto conf = std::get<0>(cf);
    auto cidx = std::get<1>(cf);
    auto idx  = torch::nonzero(conf.gt(0.05f)).flatten();   // low thr for PR curve
    torch::Tensor gt_targets = ex.targets.to(torch::kCPU).contiguous();
    if (G > 0) {
      auto ta = gt_targets.accessor<float, 2>();
      for (int g = 0; g < G; ++g) gt_count[(int)ta[g][0]]++;
    }
    if (idx.numel() == 0 || G == 0) continue;
    auto box      = p.slice(1, 0, 4).index_select(0, idx).contiguous();
    auto a_cpu    = ang.to(torch::kCPU)[0].index_select(0, idx).contiguous();
    auto cidx_sel = cidx.index_select(0, idx).contiguous();
    auto sel_conf = conf.index_select(0, idx).contiguous();
    const int k = (int)idx.numel();
    n_pred += k;
    auto a_b = box.accessor<float, 2>();
    auto a_a = a_cpu.accessor<float, 1>();
    auto a_l = cidx_sel.accessor<int64_t, 1>();
    auto cfa = sel_conf.accessor<float, 1>();
    auto a_t = gt_targets.accessor<float, 2>();

    // Pre-build rotated rects.
    std::vector<cv::RotatedRect> prr(k), grr(G);
    for (int j = 0; j < k; ++j) {
      float cx = (a_b[j][0] + a_b[j][2]) / 2, cy = (a_b[j][1] + a_b[j][3]) / 2;
      float w  = a_b[j][2] - a_b[j][0],       h  = a_b[j][3] - a_b[j][1];
      prr[j] = cv::RotatedRect({cx, cy}, {w, h}, a_a[j] * 180.f / (float)M_PI);
    }
    for (int g = 0; g < G; ++g)
      grr[g] = cv::RotatedRect({a_t[g][1], a_t[g][2]}, {a_t[g][3], a_t[g][4]},
                               a_t[g][5] * 180.f / (float)M_PI);

    // Greedy match predictions (conf desc) to best unmatched same-class GT.
    auto order = std::get<1>(sel_conf.sort(/*dim=*/0, /*descending=*/true)).contiguous();
    auto oa = order.accessor<int64_t, 1>();
    std::vector<bool> used(G, false);
    for (int oi = 0; oi < k; ++oi) {
      int j = (int)oa[oi];
      int c = (int)a_l[j];
      float best = 0.0f; int bg = -1;
      for (int g = 0; g < G; ++g) {
        if (used[g] || (int)a_t[g][0] != c) continue;
        std::vector<cv::Point2f> inter;
        if (cv::rotatedRectangleIntersection(grr[g], prr[j], inter) ==
                cv::INTERSECT_NONE || inter.size() < 3)
          continue;
        double a_int = cv::contourArea(inter);
        double a_uni = (double)grr[g].size.area() + (double)prr[j].size.area() - a_int;
        float iou = (a_uni > 0) ? (float)(a_int / a_uni) : 0.0f;
        if (iou > best) { best = iou; bg = g; }
      }
    int tp = (best > 0.5f && bg >= 0) ? 1 : 0;
      if (tp) { used[bg] = true; ++n_matched; }
      det_by_class[c].push_back({cfa[j], tp});
    }
  }
  // Per-class AP@0.5; mAP over classes with GT.
  double sum_ap = 0.0; int n_cls = 0;
  for (auto& [c, cnt] : gt_count) {
    if (cnt <= 0) continue;
    auto& d = det_by_class[c];
    std::sort(d.begin(), d.end(),
              [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                return a.first > b.first;
              });
    double tp = 0, fp = 0, ap = 0, prev_r = 0;
    for (auto& e : d) {
      if (e.second) tp += 1; else fp += 1;
      double recall = tp / cnt, precision = tp / (tp + fp);
      ap += (recall - prev_r) * precision;
      prev_r = recall;
    }
    sum_ap += ap; n_cls++;
  }
  OBBValResult r;
  r.n_pred = n_pred; r.n_gt = n_gt; r.n_matched = n_matched;
  r.map_50 = n_cls ? sum_ap / n_cls : 0.0;
  return r;
}

// Explicit instantiations — Yolo8 and Yolo11 task model holders.
template void train_pose_t<models::Yolo8Pose>(
    models::Yolo8Pose, const PoseDataset&, const PoseDataset*, PoseTrainConfig);
template void train_pose_t<models::Yolo11Pose>(
    models::Yolo11Pose, const PoseDataset&, const PoseDataset*, PoseTrainConfig);
template PoseValResult validate_pose_t<models::Yolo8Pose>(
    models::Yolo8Pose&, const PoseDataset&, torch::Device);
template PoseValResult validate_pose_t<models::Yolo11Pose>(
    models::Yolo11Pose&, const PoseDataset&, torch::Device);
template void train_obb_t<models::Yolo8OBB>(
    models::Yolo8OBB, const OBBDataset&, const OBBDataset*, OBBTrainConfig);
template void train_obb_t<models::Yolo11OBB>(
    models::Yolo11OBB, const OBBDataset&, const OBBDataset*, OBBTrainConfig);
template OBBValResult validate_obb_t<models::Yolo8OBB>(
    models::Yolo8OBB&, const OBBDataset&, torch::Device);
template OBBValResult validate_obb_t<models::Yolo11OBB>(
    models::Yolo11OBB&, const OBBDataset&, torch::Device);

}  // namespace yolocpp::tasks
