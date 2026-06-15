#include "yolocpp/tasks/segment_train.hpp"

#include "yolocpp/inference/trt_task_eval.hpp"  // TrtSegModel (TRT-backed val)
#include "yolocpp/serialization/pt_save.hpp"     // save_module_state_dict

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/optim.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/task_predictors.hpp"
#include "yolocpp/losses/yolo8_loss.hpp"

namespace fs = std::filesystem;

namespace yolocpp::tasks {

namespace {

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

// Each label line: "cls cx cy w h [x1 y1 x2 y2 ... xN yN]". Polygon vertices
// are absent for pure-detect labels, but we treat them as a tight rectangle
// in that case (so segment training tolerates detect-style labels).
struct SegLabel {
  int                cls;
  std::array<float,4> bbox_cxcywh;     // normalized
  std::vector<float> poly_xy;          // normalized, possibly empty
};

std::vector<SegLabel> parse_seg_labels(const std::string& path) {
  std::vector<SegLabel> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    SegLabel l;
    iss >> l.cls >> l.bbox_cxcywh[0] >> l.bbox_cxcywh[1]
        >> l.bbox_cxcywh[2] >> l.bbox_cxcywh[3];
    float x;
    while (iss >> x) l.poly_xy.push_back(x);
    out.push_back(std::move(l));
  }
  return out;
}

}  // anonymous namespace

SegDataset::SegDataset(std::string root, std::string split, int imgsz,
                       std::vector<std::string> names, bool augment)
    : names_(std::move(names)), imgsz_(imgsz), augment_(augment) {
  fs::path img_dir = fs::path(root) / "images" / split;
  paths_ = list_imgs(img_dir);
  for (auto& p : paths_) lbl_paths_.push_back(label_path_for(p));
  if (paths_.empty())
    throw std::runtime_error("no images at " + img_dir.string());
}

SegExample SegDataset::get(std::size_t idx, uint64_t seed) const {
  SegExample ex;
  ex.img_path = paths_[idx];
  cv::Mat bgr = cv::imread(ex.img_path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("cannot read " + ex.img_path);
  ex.orig_w = bgr.cols; ex.orig_h = bgr.rows;

  auto lb = inference::letterbox(bgr, imgsz_);
  ex.gain = lb.gain; ex.pad_x = lb.pad_x; ex.pad_y = lb.pad_y;

  // Decide on horizontal flip.
  std::mt19937 rng(seed ? seed : (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  bool flip = false;
  if (augment_) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    flip = u(rng) < 0.5f;
    if (flip) cv::flip(lb.img, lb.img, 1);
  }
  ex.img = inference::image_to_tensor(lb.img);

  // Build per-instance masks (full-res, imgsz × imgsz uint8).
  auto labels = parse_seg_labels(lbl_paths_[idx]);
  if (labels.empty()) {
    ex.targets = torch::zeros({0, 5}, torch::kFloat32);
    ex.masks   = torch::zeros({0, imgsz_, imgsz_}, torch::kUInt8);
    return ex;
  }

  std::vector<float> tgt_rows;
  std::vector<cv::Mat> masks;
  masks.reserve(labels.size());

  for (const auto& l : labels) {
    // Map bbox to letterboxed pixel coords.
    float cx = l.bbox_cxcywh[0] * (float)ex.orig_w;
    float cy = l.bbox_cxcywh[1] * (float)ex.orig_h;
    float w  = l.bbox_cxcywh[2] * (float)ex.orig_w;
    float h  = l.bbox_cxcywh[3] * (float)ex.orig_h;
    cx = (float)(cx * lb.gain + lb.pad_x);
    cy = (float)(cy * lb.gain + lb.pad_y);
    w  = (float)(w  * lb.gain);
    h  = (float)(h  * lb.gain);
    if (flip) cx = (float)imgsz_ - 1.f - cx;
    tgt_rows.push_back((float)l.cls);
    tgt_rows.push_back(cx); tgt_rows.push_back(cy);
    tgt_rows.push_back(w);  tgt_rows.push_back(h);

    // Rasterize polygon mask. If no polygon, fall back to bbox.
    cv::Mat m(imgsz_, imgsz_, CV_8UC1, cv::Scalar(0));
    if (l.poly_xy.size() >= 6) {
      std::vector<cv::Point> pts;
      for (size_t k = 0; k + 1 < l.poly_xy.size(); k += 2) {
        float px = l.poly_xy[k]   * (float)ex.orig_w;
        float py = l.poly_xy[k+1] * (float)ex.orig_h;
        px = (float)(px * lb.gain + lb.pad_x);
        py = (float)(py * lb.gain + lb.pad_y);
        if (flip) px = (float)imgsz_ - 1.f - px;
        pts.emplace_back((int)std::round(px), (int)std::round(py));
      }
      std::vector<std::vector<cv::Point>> conts{pts};
      cv::fillPoly(m, conts, cv::Scalar(1));
    } else {
      cv::Rect r((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h);
      r &= cv::Rect(0, 0, imgsz_, imgsz_);
      if (r.width > 0 && r.height > 0) m(r).setTo(1);
    }
    masks.push_back(m);
  }

  ex.targets = torch::from_blob(tgt_rows.data(),
                                {(int64_t)labels.size(), 5},
                                torch::kFloat32).clone();
  ex.masks = torch::zeros({(int64_t)labels.size(), imgsz_, imgsz_},
                          torch::kUInt8);
  for (size_t i = 0; i < masks.size(); ++i) {
    auto m_t = torch::from_blob(masks[i].data,
                                {imgsz_, imgsz_}, torch::kUInt8).clone();
    ex.masks.index_put_({(int64_t)i}, m_t);
  }
  return ex;
}

SegDataset::Batch SegDataset::sample_batch(std::size_t bsz,
                                           std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, paths_.size() - 1);
  std::vector<torch::Tensor> imgs, tgts, masks;
  std::vector<SegExample> exs;
  for (size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), (uint64_t)rng() << 32 | i);
    imgs.push_back(ex.img);
    if (ex.targets.size(0) > 0) {
      auto bcol = torch::full({ex.targets.size(0), 1}, (double)i, torch::kFloat32);
      tgts.push_back(torch::cat({bcol, ex.targets}, 1));
      masks.push_back(ex.masks);
    }
    exs.push_back(std::move(ex));
  }
  Batch b;
  b.imgs = torch::stack(imgs, 0);
  b.targets = tgts.empty() ? torch::zeros({0, 6}, torch::kFloat32)
                           : torch::cat(tgts, 0);
  b.masks   = masks.empty() ? torch::zeros({0, imgsz_, imgsz_}, torch::kUInt8)
                            : torch::cat(masks, 0);
  b.examples = std::move(exs);
  return b;
}

// ─── Loss helper: detect part + mask BCE ──────────────────────────────────

namespace {

torch::Device pick_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}

// Compute mask BCE for the batch.
//   coefs: [B, nm, A]
//   protos: [B, nm, h_p, w_p]
//   targets: [M, 6] — (b_idx, cls, cx, cy, w, h) in input pixel coords
//   masks: [M, H_in, W_in] uint8
// Strategy: assign each ground-truth instance to its single closest anchor
// (by center proximity) at the smallest stride, decode that anchor's predicted
// mask, BCE against the (resized to proto resolution) GT mask.
torch::Tensor compute_mask_loss(const torch::Tensor& coefs,
                                const torch::Tensor& protos,
                                const torch::Tensor& targets,
                                const torch::Tensor& masks,
                                int imgsz, int nm,
                                const std::vector<double>& strides) {
  if (targets.size(0) == 0)
    return torch::zeros({}, coefs.options());
  // Per-target: pick a single anchor at stride[0] (smallest stride) at
  // (cx, cy). Anchors are at ((j+0.5)*s, (i+0.5)*s).
  double s0 = strides[0];
  int    feat_h = imgsz / (int)s0;
  int    feat_w = imgsz / (int)s0;
  int    A_lvl  = feat_h * feat_w;
  int    B      = (int)protos.size(0);

  auto t = targets.detach().to(torch::kCPU);
  auto a = t.accessor<float, 2>();

  std::vector<int64_t> anchor_idxs;
  std::vector<int64_t> batch_idxs;
  std::vector<int>     valid;
  anchor_idxs.reserve(t.size(0));
  batch_idxs.reserve(t.size(0));
  for (int i = 0; i < (int)t.size(0); ++i) {
    int b = (int)a[i][0];
    if (b < 0 || b >= B) continue;
    float cx = a[i][2], cy = a[i][3];
    int xi = std::clamp((int)std::floor(cx / s0), 0, feat_w - 1);
    int yi = std::clamp((int)std::floor(cy / s0), 0, feat_h - 1);
    anchor_idxs.push_back(yi * feat_w + xi);
    batch_idxs.push_back(b);
    valid.push_back(i);
  }
  if (valid.empty()) return torch::zeros({}, coefs.options());

  auto anc_t = torch::tensor(anchor_idxs,
                             torch::TensorOptions().device(coefs.device())
                                                   .dtype(torch::kLong));
  auto bat_t = torch::tensor(batch_idxs,
                             torch::TensorOptions().device(coefs.device())
                                                   .dtype(torch::kLong));

  // Gather coefficients: coefs[b, :, anchor_idx] for each (b, anchor_idx).
  // Note: the smallest stride's anchors are the *first* A_lvl entries in A.
  auto coefs_lvl0 = coefs.slice(2, 0, A_lvl);                          // [B, nm, A_lvl]
  auto coefs_pos  = coefs_lvl0.index({bat_t, torch::indexing::Slice(),
                                       anc_t});                        // [P, nm]
  // coefs_pos shape after fancy indexing is [P, nm].
  auto protos_pos = protos.index({bat_t});                             // [P, nm, h_p, w_p]
  // Predicted masks: [P, h_p, w_p]
  auto h_p = protos_pos.size(2), w_p = protos_pos.size(3);
  auto pflat = protos_pos.reshape({-1, nm, h_p * w_p});                // [P, nm, h_p*w_p]
  auto pred  = torch::matmul(coefs_pos.unsqueeze(1), pflat)             // [P, 1, h_p*w_p]
                   .squeeze(1)
                   .reshape({-1, h_p, w_p});

  // GT masks: pick the rows in `valid`, downsample to (h_p, w_p), float.
  std::vector<torch::Tensor> gts;
  gts.reserve(valid.size());
  // GT masks are stored as 0/1 (fillPoly value 1) — NOT 0/255. Dividing by
  // 255 here collapsed every GT to ~0.004 → `(gt > 0.5)` made the whole target
  // all-zero → the mask head was trained to predict empty masks (the train-side
  // twin of the validator /255 bug). Keep them 0/1.
  auto masks_cpu = masks.to(torch::kFloat32);
  for (int i : valid) gts.push_back(masks_cpu[i]);
  auto gt = torch::stack(gts, 0).unsqueeze(1);                         // [P, 1, H, W]
  gt = gt.clamp(0.0, 1.0);
  gt = torch::nn::functional::interpolate(
      gt,
      torch::nn::functional::InterpolateFuncOptions()
          .size(std::vector<int64_t>{h_p, w_p})
          .mode(torch::kBilinear)
          .align_corners(false));
  gt = gt.squeeze(1).to(coefs.device());                                // [P, h_p, w_p]
  gt = (gt > 0.5f).to(coefs.dtype());

  return torch::nn::functional::binary_cross_entropy_with_logits(
      pred, gt,
      torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
          .reduction(torch::kMean));
}

}  // anonymous namespace

template <typename M>
void train_segment_t(M model,
                   const SegDataset& train,
                   const SegDataset* val,
                   SegTrainConfig cfg) {
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

  // Detect-side loss.
  losses::LossConfig lc;
  lc.nc = train.num_classes();
  losses::V8DetectionLoss det_loss(lc);

  std::mt19937 rng(0x9E3779B9u);
  size_t n     = train.size();
  int    steps = std::max<int>(1, (int)(n / cfg.batch_size));
  int    total = steps * cfg.epochs;
  int    warmup = std::max(50, steps * cfg.warmup_epochs);
  std::cout << "[seg-train] " << n << " imgs, " << steps << " steps/epoch, "
            << "device=" << device << "\n";

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto t0 = std::chrono::steady_clock::now();
    double sum_d = 0, sum_m = 0;
    for (int step = 0; step < steps; ++step) {
      int gstep = epoch * steps + step;
      double scale;
      if (gstep < warmup)
        scale = (double)(gstep + 1) / (double)warmup;
      else {
        double t = (double)(gstep - warmup) /
                   std::max(1.0, (double)(total - warmup));
        scale = cfg.lrf + 0.5 * (1.0 - cfg.lrf) * (1.0 + std::cos(M_PI * t));
      }
      auto& gs = optim.param_groups();
      static_cast<torch::optim::SGDOptions&>(gs[0].options()).lr(cfg.lr0 * scale);

      auto b = train.sample_batch(cfg.batch_size, rng);
      auto x = b.imgs.to(device);
      auto tgt = b.targets.to(device);
      auto masks = b.masks;  // kept on CPU until needed in compute_mask_loss

      // Training forward (§5): forward_train_seg returns the RAW per-level
      // detect features (for the box/cls/dfl loss) plus the mask coefs +
      // prototypes (for the mask loss) in one pass. Previously the detect
      // loss was skipped (det_total = 0) because the feats weren't exposed —
      // segment training optimised mask loss only, which left box/cls/dfl
      // un-trained.
      auto [feats, coefs, protos] = model->forward_train_seg(x);
      auto* seg_head =
          model->model[model->model->size() - 1]->template as<models::SegmentImpl>();

      // Detect loss (box + cls + dfl) on the raw per-level features —
      // identical to the detect path (V8DetectionLoss).
      auto det = det_loss(feats, tgt, model->stride, cfg.imgsz);
      torch::Tensor det_total = det.total;

      // Mask loss from coefficients + prototypes.
      torch::Tensor mask_loss = compute_mask_loss(
          coefs, protos, tgt, masks, cfg.imgsz, seg_head->nm,
          model->stride);

      auto loss = det_total + mask_loss * cfg.mask_gain;
      optim.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model->parameters(), 10.0);
      optim.step();

      sum_d += det_total.template item<double>();
      sum_m += mask_loss.template item<double>();
      if (gstep % cfg.log_every == 0)
        std::cout << "[seg-train] e=" << epoch << " s=" << step
                  << " det=" << det_total.template item<double>()
                  << " mask=" << mask_loss.template item<double>() << "\n";
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[seg-train] epoch " << epoch
              << " avg_det=" << (sum_d / steps)
              << " avg_mask=" << (sum_m / steps)
              << " in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";
  }
  auto ckpt = fs::path(cfg.save_dir) / "last.pt";
  serialization::save_module_state_dict(*model, ckpt.string());
  std::cout << "[seg-train] saved → " << ckpt << "\n";
}

template <typename M>
SegValResult validate_segment_t(M& model,
                               const SegDataset& dataset,
                               torch::Device device) {
  model->to(device);
  model->eval();
  // Real per-class mask AP@0.5. Per image: decode the predicted instance
  // masks, CROP each to its predicted box (the upstream `crop_mask` step —
  // without it a mask spans the whole proto activation and rarely reaches
  // IoU 0.5), then greedily match preds (highest-conf first) to unmatched GT
  // of the SAME class by mask IoU > 0.5 → TP/FP. AP@0.5 = area under the
  // per-class precision/recall step curve; mAP averages classes with GT.
  // (Previously this divided the 0/1 GT masks by 255 → all-empty GT → 0
  //  matches → mAP always 0, and never cropped predicted masks.)
  const int sz = dataset.imgsz();
  const int max_det = 100;
  std::map<int, std::vector<std::pair<float, int>>> det_by_class;  // cls → (conf, tp)
  std::map<int, int> gt_count;
  int total_pred = 0, total_gt = 0;

  for (size_t i = 0; i < dataset.size(); ++i) {
    auto ex = dataset.get(i, /*seed=*/i + 1);
    auto x  = ex.img.unsqueeze(0).to(device);
    torch::Tensor pred, coefs, protos;
    {
      torch::NoGradGuard ng;
      auto out = model->forward_eval(x);
      pred = std::get<0>(out); coefs = std::get<1>(out); protos = std::get<2>(out);
    }

    // GT masks (0/1 at full input res) + per-instance class.
    auto gt = ex.masks.to(torch::kCPU).to(torch::kFloat32).gt(0.5f).to(torch::kFloat32);
    const int G = (int)gt.size(0);
    total_gt += G;
    torch::Tensor gt_cls;
    if (G > 0) {
      gt_cls = ex.targets.slice(1, 0, 1).to(torch::kCPU).reshape({-1}).contiguous();
      auto gca = gt_cls.accessor<float, 1>();
      for (int g = 0; g < G; ++g) gt_count[(int)gca[g]]++;
    }

    // Decode predictions: [1, 4+nc, A] → [A, 4+nc] (xyxy px + sigmoid cls).
    auto p     = pred.transpose(1, 2).contiguous().to(torch::kCPU)[0];
    auto boxes = p.slice(1, 0, 4);
    auto clsm  = p.slice(1, 4, p.size(1));
    auto best  = clsm.max(1);
    auto conf  = std::get<0>(best);
    auto cid   = std::get<1>(best);
    auto idx   = torch::nonzero(conf.gt(0.05f)).flatten();   // low thr for the PR curve
    if (idx.numel() == 0 || G == 0) continue;
    // Cap to the top-`max_det` by confidence (COCO-style).
    if (idx.numel() > max_det) {
      auto top = std::get<1>(conf.index_select(0, idx).sort(0, true)).slice(0, 0, max_det);
      idx = idx.index_select(0, top).contiguous();
    }

    auto sel_coefs = coefs.transpose(1, 2)[0].to(torch::kCPU).index_select(0, idx);  // [k,nm]
    auto sel_box   = boxes.index_select(0, idx).contiguous();                        // [k,4]
    auto sel_conf  = conf.index_select(0, idx).contiguous();                         // [k]
    auto sel_cls   = cid.index_select(0, idx).contiguous();                          // [k]
    const int k = (int)idx.numel();
    total_pred += k;

    // proto-masks → sigmoid → resize to (sz,sz) → threshold 0.5.
    auto p_cpu  = protos.to(torch::kCPU)[0];
    auto h_p = p_cpu.size(1), w_p = p_cpu.size(2);
    auto p_flat = p_cpu.reshape({p_cpu.size(0), h_p * w_p});
    auto m = sel_coefs.matmul(p_flat).reshape({-1, h_p, w_p}).sigmoid();
    m = torch::nn::functional::interpolate(
            m.unsqueeze(1),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{(int64_t)sz, (int64_t)sz})
                .mode(torch::kBilinear).align_corners(false))
            .squeeze(1).gt(0.5f).to(torch::kFloat32);          // [k, sz, sz]

    // crop_mask: zero everything outside each predicted box.
    {
      auto ba = sel_box.accessor<float, 2>();
      for (int j = 0; j < k; ++j) {
        int x1 = std::clamp((int)std::floor(ba[j][0]), 0, sz);
        int y1 = std::clamp((int)std::floor(ba[j][1]), 0, sz);
        int x2 = std::clamp((int)std::ceil(ba[j][2]),  0, sz);
        int y2 = std::clamp((int)std::ceil(ba[j][3]),  0, sz);
        auto mj = m[j];
        if (y1 > 0)  mj.slice(0, 0, y1).zero_();
        if (y2 < sz) mj.slice(0, y2, sz).zero_();
        if (x1 > 0)  mj.slice(1, 0, x1).zero_();
        if (x2 < sz) mj.slice(1, x2, sz).zero_();
      }
    }

    // Pairwise mask IoU [k, G].
    auto m_flat = m.reshape({k, -1});
    auto g_flat = gt.reshape({G, -1});
    auto inter  = m_flat.matmul(g_flat.t());
    auto m_a    = m_flat.sum(1, true);
    auto g_a    = g_flat.sum(1, true).t();
    auto iou    = (inter / (m_a + g_a - inter + 1e-7f)).contiguous();   // [k, G]

    // Greedy match: preds by conf desc → best unmatched same-class GT, IoU>0.5.
    auto order = std::get<1>(sel_conf.sort(/*dim=*/0, /*descending=*/true)).contiguous();
    auto oa  = order.accessor<int64_t, 1>();
    auto ia  = iou.accessor<float, 2>();
    auto sca = sel_cls.accessor<int64_t, 1>();
    auto cfa = sel_conf.accessor<float, 1>();
    auto gca = gt_cls.accessor<float, 1>();
    std::vector<bool> gt_used(G, false);
    for (int oi = 0; oi < k; ++oi) {
      int j = (int)oa[oi];
      int c = (int)sca[j];
      float best_iou = 0.0f; int best_g = -1;
      for (int g = 0; g < G; ++g) {
        if (gt_used[g] || (int)gca[g] != c) continue;
        if (ia[j][g] > best_iou) { best_iou = ia[j][g]; best_g = g; }
      }
      int tp = (best_iou > 0.5f && best_g >= 0) ? 1 : 0;
      if (tp) gt_used[best_g] = true;
      det_by_class[c].push_back({cfa[j], tp});
    }
  }

  // Per-class AP@0.5 (area under the recall→precision step curve); mAP over
  // classes that have at least one GT instance.
  double sum_ap = 0.0; int n_cls = 0;
  for (auto& [c, cnt] : gt_count) {
    if (cnt <= 0) continue;
    auto& dets = det_by_class[c];
    std::sort(dets.begin(), dets.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
              });
    double tp = 0, fp = 0, ap = 0, prev_r = 0;
    for (auto& d : dets) {
      if (d.second) tp += 1; else fp += 1;
      double recall    = tp / cnt;
      double precision = tp / (tp + fp);
      ap += (recall - prev_r) * precision;
      prev_r = recall;
    }
    sum_ap += ap; n_cls++;
  }

  SegValResult r;
  r.map_50          = n_cls ? sum_ap / n_cls : 0.0;
  r.n_predictions   = total_pred;
  r.n_ground_truths = total_gt;
  return r;
}

// Explicit instantiations — Yolo8Segment + Yolo11Segment.
template void train_segment_t<models::Yolo8Segment>(
    models::Yolo8Segment, const SegDataset&, const SegDataset*, SegTrainConfig);
template void train_segment_t<models::Yolo11Segment>(
    models::Yolo11Segment, const SegDataset&, const SegDataset*, SegTrainConfig);
template SegValResult validate_segment_t<models::Yolo8Segment>(
    models::Yolo8Segment&, const SegDataset&, torch::Device);
template SegValResult validate_segment_t<models::Yolo11Segment>(
    models::Yolo11Segment&, const SegDataset&, torch::Device);
template void train_segment_t<models::Yolo12Segment>(
    models::Yolo12Segment, const SegDataset&, const SegDataset*, SegTrainConfig);
template SegValResult validate_segment_t<models::Yolo12Segment>(
    models::Yolo12Segment&, const SegDataset&, torch::Device);
template void train_segment_t<models::Yolo13Segment>(
    models::Yolo13Segment, const SegDataset&, const SegDataset*, SegTrainConfig);
template SegValResult validate_segment_t<models::Yolo13Segment>(
    models::Yolo13Segment&, const SegDataset&, torch::Device);
// TRT-backed validation (per-format benchmark mAP) reuses the same metric.
template SegValResult validate_segment_t<inference::TrtSegModel>(
    inference::TrtSegModel&, const SegDataset&, torch::Device);

}  // namespace yolocpp::tasks
