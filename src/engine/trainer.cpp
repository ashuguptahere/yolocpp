#include "yolocpp/engine/trainer.hpp"

#include <torch/optim.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "yolocpp/engine/ddp.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/losses/yolo26_loss.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::engine {

namespace fs = std::filesystem;

// Pick a fresh "<base><N>" directory: returns base if it doesn't exist,
// else base2, base3, … — Ultralytics convention.
static std::string increment_save_dir(const std::string& base) {
  if (!fs::exists(base)) return base;
  for (int i = 2; i < 10000; ++i) {
    auto p = base + std::to_string(i);
    if (!fs::exists(p)) return p;
  }
  return base + "_overflow";
}

static torch::Device pick_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  // Multi-GPU spec like "cuda:0,1" is launched via scripts/launch_ddp.sh,
  // which spawns one process per GPU and sets RANK/LOCAL_RANK env vars.
  // If we see a comma here in a single process, the user invoked the binary
  // directly — point them at the launcher.
  if (s.find(',') != std::string::npos) {
    throw std::runtime_error(
        "multi-GPU training (device='" + s + "') needs the launcher: "
        "`scripts/launch_ddp.sh <ngpus> task=detect mode=train ...` "
        "(spawns one process per GPU and sets RANK/LOCAL_RANK).");
  }
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}

namespace {

// Build three parameter groups (decay-conv, no-decay-bias-bn, BN-weight-no-decay)
// matching Ultralytics' SGD setup.
struct ParamGroups {
  std::vector<at::Tensor> decay;
  std::vector<at::Tensor> bias;
  std::vector<at::Tensor> bn;
};

ParamGroups split_params(torch::nn::Module& m) {
  ParamGroups pg;
  for (auto& kv : m.named_parameters()) {
    const auto& name = kv.key();
    auto& p = kv.value();
    if (!p.requires_grad()) continue;
    if (name.size() >= 5 && name.substr(name.size() - 5) == ".bias") {
      pg.bias.push_back(p);
    } else if (name.find(".bn.") != std::string::npos &&
               (name.size() >= 7 && name.substr(name.size() - 7) == ".weight")) {
      pg.bn.push_back(p);
    } else {
      pg.decay.push_back(p);
    }
  }
  return pg;
}

// Loss-type trait — selects the right loss for each model holder.
//
// Default (v8/v5/v11) → V8DetectionLoss (DFL-based).
// Specialised for Yolo26Detect → Yolo26Loss (DFL-free, STAL + ProgLoss).
//
// Both losses expose:
//   - a configurable nc field at construction
//   - operator()(feats, tgt, strides, imgsz [, progress]) → LossOutput-like
// LossOutput-like must have .total / .box / .cls / .dfl scalar tensors.
template <typename M>
struct LossTraits {
  using LossT   = losses::V8DetectionLoss;
  using OutputT = losses::LossOutput;
  static LossT make(int nc) {
    losses::LossConfig c; c.nc = nc; c.reg_max = 16;
    return LossT(c);
  }
  static OutputT compute(const LossT& l,
                         const std::vector<torch::Tensor>& feats,
                         const torch::Tensor& tgt,
                         const std::vector<double>& strides,
                         int imgsz, double /*progress*/) {
    return l(feats, tgt, strides, imgsz);
  }
};

template <>
struct LossTraits<models::Yolo26Detect> {
  using LossT   = losses::Yolo26Loss;
  using OutputT = losses::Yolo26LossOutput;
  static LossT make(int nc) {
    losses::Yolo26LossConfig c; c.nc = nc;
    return LossT(c);
  }
  static OutputT compute(const LossT& l,
                         const std::vector<torch::Tensor>& feats,
                         const torch::Tensor& tgt,
                         const std::vector<double>& strides,
                         int imgsz, double progress) {
    return l(feats, tgt, strides, imgsz, progress);
  }
};

// Cosine-with-linear-warmup LR schedule. Returns lr scale ∈ [0, 1].
double lr_scale(int epoch_step, int warmup_steps, int total_steps,
                double lrf) {
  if (epoch_step < warmup_steps) {
    // Linear ramp from 0 → 1
    return (double)(epoch_step + 1) / (double)warmup_steps;
  }
  double t = (double)(epoch_step - warmup_steps) /
             std::max(1.0, (double)(total_steps - warmup_steps));
  // cosine: 1 → lrf
  return lrf + 0.5 * (1.0 - lrf) * (1.0 + std::cos(M_PI * t));
}

}  // anonymous namespace

template <typename M>
TrainerT<M>::TrainerT(M model, datasets::YoloDataset train, TrainConfig cfg)
    : model_(std::move(model)),
      ema_(M(model_->scale, model_->nc)),
      train_(std::move(train)),
      cfg_(std::move(cfg)),
      device_(pick_device(cfg_.device)) {
  // Init EMA from current model weights (deep copy params + buffers).
  torch::NoGradGuard ng;
  auto src_p = model_->named_parameters();
  auto dst_p = ema_->named_parameters();
  for (auto& kv : src_p) {
    auto* d = dst_p.find(kv.key());
    if (d) d->copy_(kv.value().detach());
  }
  auto src_b = model_->named_buffers();
  auto dst_b = ema_->named_buffers();
  for (auto& kv : src_b) {
    auto* d = dst_b.find(kv.key());
    if (d) d->copy_(kv.value().detach());
  }
  // EMA copies stride too.
  ema_->stride = model_->stride;

  for (auto& p : ema_->parameters()) p.set_requires_grad(false);
}

template <typename M>
void TrainerT<M>::ema_update(double decay) {
  torch::NoGradGuard ng;
  auto m_params = model_->named_parameters();
  auto e_params = ema_->named_parameters();
  for (auto& kv : m_params) {
    auto* dst = e_params.find(kv.key());
    if (!dst) continue;
    dst->mul_(decay).add_(kv.value().detach(), 1.0 - decay);
  }
  // Buffers (BN running stats) follow the model directly.
  auto m_buf = model_->named_buffers();
  auto e_buf = ema_->named_buffers();
  for (auto& kv : m_buf) {
    auto* dst = e_buf.find(kv.key());
    if (!dst) continue;
    dst->copy_(kv.value().detach());
  }
}

template <typename M>
void TrainerT<M>::run() {
  // Initialise DDP from env vars; no-op when WORLD_SIZE is unset.
  auto ddp = init_ddp_from_env();
  if (ddp.active) {
    // Pin this process to its assigned local GPU.
    device_ = torch::Device(torch::kCUDA, ddp.local_rank);
    if (is_rank0(ddp)) {
      std::cout << "[trainer] DDP active: world_size=" << ddp.world_size
                << " local_rank=" << ddp.local_rank << "\n";
    }
  }

  // Only rank 0 picks the run dir; broadcast it to others so they all log
  // into the same place. (For our simple trainer we just have rank 0 write
  // I/O; non-rank-0 just trains without writing.)
  if (is_rank0(ddp)) {
    cfg_.save_dir = increment_save_dir(cfg_.save_dir);
    fs::create_directories(cfg_.save_dir);
    std::cout << "[trainer] save_dir = " << cfg_.save_dir << "\n";
  }
  barrier(ddp);

  // Dump the run's input args for reproducibility — Ultralytics-shape
  // args.yaml so existing tooling can consume it. The list of keys and
  // default values follows Ultralytics ≥ 8.3 (see ultralytics/cfg/default.yaml).
  if (is_rank0(ddp)) {
    std::ofstream y((fs::path(cfg_.save_dir) / "args.yaml").string());

    // Helper: was this key supplied on the CLI? If so, use that value;
    // otherwise emit the Ultralytics default.
    auto kv_lookup = [&](const std::string& k) -> std::string {
      for (const auto& [ck, cv] : cfg_.args_for_yaml)
        if (ck == k) return cv;
      return "";
    };
    auto emit = [&](const std::string& k, const std::string& dflt) {
      auto v = kv_lookup(k);
      const auto& out = !v.empty() ? v : dflt;
      // YAML quoting: only quote strings that contain '/', ':', or spaces.
      bool needs_quote = out.find(' ') != std::string::npos ||
                         out.find(':') != std::string::npos;
      y << k << ": " << (needs_quote ? "\"" : "") << out
                     << (needs_quote ? "\"" : "") << "\n";
    };

    // ISO-8601 timestamp.
    auto t  = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(t);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));
    y << "# yolocpp training run — generated by Trainer at " << ts << "\n";
    y << "# (matches Ultralytics ≥ 8.3 default.yaml field list)\n";

    // ── Section 1: task / mode / I-O ──────────────────────────────────
    emit("task",        "detect");
    emit("mode",        "train");
    emit("model",       "");                  // weights .pt
    emit("data",        "");                  // dataset root or yaml
    emit("epochs",      std::to_string(cfg_.epochs));
    y << "time:\n";
    emit("patience",    std::to_string(cfg_.patience > 0 ? cfg_.patience : 100));
    emit("batch",       std::to_string(cfg_.batch_size));
    emit("imgsz",       std::to_string(cfg_.imgsz));
    emit("save",        "true");
    emit("save_period", "-1");
    emit("cache",       "false");
    {
      std::ostringstream ss; ss << device_;
      emit("device",      ss.str());
    }
    emit("workers",     "8");
    y << "project:\n";
    y << "name:\n";
    emit("exist_ok",    "false");
    emit("pretrained",  kv_lookup("model").empty() ? "true" : "true");
    emit("optimizer",   "auto");
    emit("verbose",     "true");
    emit("seed",        "0");
    emit("deterministic","true");
    emit("single_cls",  "false");
    emit("rect",        "false");
    emit("cos_lr",      "false");
    emit("close_mosaic","10");
    emit("resume",      "false");
    emit("amp",         "true");
    emit("fraction",    "1.0");
    emit("profile",     "false");
    y << "freeze:\n";
    emit("multi_scale", "false");

    // ── Section 2: segmentation/keypoint/box overrides ────────────────
    emit("overlap_mask","true");
    emit("mask_ratio",  "4");
    emit("dropout",     "0.0");
    emit("val",         "true");
    emit("split",       "val");
    emit("save_json",   "false");
    emit("save_hybrid", "false");
    y << "conf:\n";
    emit("iou",         "0.7");
    emit("max_det",     "300");
    emit("half",        "false");
    emit("dnn",         "false");
    emit("plots",       "true");

    // ── Section 3: predict / source / display ─────────────────────────
    y << "source:\n";
    emit("vid_stride",  "1");
    emit("stream_buffer","false");
    emit("visualize",   "false");
    emit("augment",     "false");
    emit("agnostic_nms","false");
    y << "classes:\n";
    emit("retina_masks","false");
    y << "embed:\n";
    emit("show",        "false");
    emit("save_frames", "false");
    emit("save_txt",    "false");
    emit("save_conf",   "false");
    emit("save_crop",   "false");
    emit("show_labels", "true");
    emit("show_conf",   "true");
    emit("show_boxes",  "true");
    y << "line_width:\n";

    // ── Section 4: export ─────────────────────────────────────────────
    emit("format",      "torchscript");
    emit("keras",       "false");
    emit("optimize",    "false");
    emit("int8",        "false");
    emit("dynamic",     "false");
    emit("simplify",    "true");
    y << "opset:\n";
    y << "workspace:\n";
    emit("nms",         "false");

    // ── Section 5: hyperparameters ────────────────────────────────────
    {
      std::ostringstream ss; ss << cfg_.lr0;
      emit("lr0",         ss.str());
    }
    {
      std::ostringstream ss; ss << cfg_.lrf;
      emit("lrf",         ss.str());
    }
    {
      std::ostringstream ss; ss << cfg_.momentum;
      emit("momentum",    ss.str());
    }
    {
      std::ostringstream ss; ss << cfg_.weight_decay;
      emit("weight_decay", ss.str());
    }
    emit("warmup_epochs", std::to_string(cfg_.warmup_epochs) + ".0");
    {
      std::ostringstream ss; ss << cfg_.warmup_momentum;
      emit("warmup_momentum", ss.str());
    }
    {
      std::ostringstream ss; ss << cfg_.warmup_bias_lr;
      emit("warmup_bias_lr", ss.str());
    }
    emit("box",         "7.5");
    emit("cls",         "0.5");
    emit("dfl",         "1.5");
    emit("pose",        "12.0");
    emit("kobj",        "1.0");
    emit("nbs",         "64");
    emit("hsv_h",       "0.015");
    emit("hsv_s",       "0.7");
    emit("hsv_v",       "0.4");
    emit("degrees",     "0.0");
    emit("translate",   "0.1");
    emit("scale",       "0.5");
    emit("shear",       "0.0");
    emit("perspective", "0.0");
    emit("flipud",      "0.0");
    emit("fliplr",      "0.5");
    emit("bgr",         "0.0");
    emit("mosaic",      "1.0");
    emit("mixup",       "0.0");
    emit("copy_paste",  "0.0");
    emit("copy_paste_mode", "flip");
    emit("auto_augment","randaugment");
    emit("erasing",     "0.4");
    emit("crop_fraction","1.0");

    // ── Section 6: tracker / cfg / save_dir ───────────────────────────
    y << "cfg:\n";
    emit("tracker",     "botsort.yaml");
    y << "save_dir: " << cfg_.save_dir << "\n";
  }

  model_->to(device_);
  ema_->to(device_);
  // Sync rank-0 weights to all ranks before the first step.
  broadcast_module(ddp, *model_);
  broadcast_module(ddp, *ema_);
  model_->train();

  auto pg = split_params(*model_);
  auto opt_options = torch::optim::SGDOptions(cfg_.lr0)
                          .momentum(cfg_.momentum)
                          .nesterov(true);

  // Group 1: decay-conv weights (with weight_decay)
  // Group 2: biases (lr=warmup_bias_lr at start)
  // Group 3: bn weights (no decay)
  std::vector<torch::optim::OptimizerParamGroup> groups;
  {
    auto g = torch::optim::OptimizerParamGroup(pg.decay);
    auto opt = std::make_unique<torch::optim::SGDOptions>(cfg_.lr0);
    opt->momentum(cfg_.momentum).weight_decay(cfg_.weight_decay).nesterov(true);
    g.set_options(std::move(opt));
    groups.push_back(std::move(g));
  }
  {
    auto g = torch::optim::OptimizerParamGroup(pg.bias);
    auto opt = std::make_unique<torch::optim::SGDOptions>(cfg_.lr0);
    opt->momentum(cfg_.momentum).weight_decay(0.0).nesterov(true);
    g.set_options(std::move(opt));
    groups.push_back(std::move(g));
  }
  {
    auto g = torch::optim::OptimizerParamGroup(pg.bn);
    auto opt = std::make_unique<torch::optim::SGDOptions>(cfg_.lr0);
    opt->momentum(cfg_.momentum).weight_decay(0.0).nesterov(true);
    g.set_options(std::move(opt));
    groups.push_back(std::move(g));
  }
  torch::optim::SGD optim(groups, opt_options);

  using Traits = LossTraits<M>;
  auto loss = Traits::make(model_->nc);

  std::mt19937 rng(0x9E3779B9u);
  size_t       n      = train_.size();
  int          steps  = std::max<int>(1, (int)(n / cfg_.batch_size));
  int          total_steps   = steps * cfg_.epochs;
  // Warmup: target = steps × warmup_epochs (Ultralytics default 3 epochs).
  // Floor at 100 steps for big datasets, but never exceed half the total —
  // otherwise on tiny datasets all of training stays in linear warmup and
  // cosine decay never kicks in.
  int          warmup_target = std::max(100, steps * cfg_.warmup_epochs);
  int          warmup_steps  = std::min(warmup_target, total_steps / 2);
  warmup_steps = std::max(1, warmup_steps);
  int          ema_step      = 0;

  std::cout << "[trainer] dataset=" << n << " imgs, "
            << steps << " steps/epoch, total=" << total_steps
            << ", device=" << device_ << "\n";

  // Helper to dump current EMA model to <save_dir>/<name>.pt in our
  // load_state_dict-compatible format.
  auto save_ckpt = [&](const std::string& name) {
    auto ckpt = fs::path(cfg_.save_dir) / name;
    std::vector<std::pair<std::string, at::Tensor>> entries;
    for (const auto& kv : ema_->named_parameters())
      entries.emplace_back(kv.key(), kv.value().detach());
    for (const auto& kv : ema_->named_buffers())
      entries.emplace_back(kv.key(), kv.value().detach());
    serialization::save_state_dict(ckpt.string(), entries);
    std::cout << "[trainer] saved → " << ckpt << "\n";
  };

  // Open results.csv (Ultralytics-compatible header).
  std::ofstream csv((fs::path(cfg_.save_dir) / "results.csv").string());
  csv << "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
         "metrics/mAP50(B),metrics/mAP50-95(B),lr0\n";
  csv << std::fixed;

  // Track best mAP across val passes (only meaningful when val_dataset set).
  double best_map = -1.0;
  int    best_epoch = -1;
  int    epochs_since_best = 0;

  for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
    auto t0 = std::chrono::steady_clock::now();
    double sum_box = 0, sum_cls = 0, sum_dfl = 0, sum_total = 0;

    for (int step = 0; step < steps; ++step) {
      int gstep = epoch * steps + step;
      double scale = lr_scale(gstep, warmup_steps, total_steps, cfg_.lrf);

      // Update LRs per group.
      // Group 0 (decay): lr = lr0 * scale
      // Group 1 (bias):  lr ramps from warmup_bias_lr → lr0 * scale during warmup
      // Group 2 (bn):    lr = lr0 * scale
      auto& gs = optim.param_groups();
      double lr_main = cfg_.lr0 * scale;
      double lr_bias = cfg_.lr0 * scale;
      if (gstep < warmup_steps) {
        double w = (double)gstep / (double)warmup_steps;
        lr_bias = cfg_.warmup_bias_lr + (lr_main - cfg_.warmup_bias_lr) * w;
      }
      static_cast<torch::optim::SGDOptions&>(gs[0].options()).lr(lr_main);
      static_cast<torch::optim::SGDOptions&>(gs[1].options()).lr(lr_bias);
      static_cast<torch::optim::SGDOptions&>(gs[2].options()).lr(lr_main);

      auto batch = train_.sample_batch(cfg_.batch_size, rng);
      auto imgs  = batch.imgs.to(device_);
      auto tgt   = batch.targets.to(device_);

      // Emit train_batchN.jpg sanity grids for the first 3 batches of the
      // first epoch — Ultralytics convention.
      if (is_rank0(ddp) && epoch == 0 && step < 3) {
        auto names = cfg_.val_dataset
                         ? cfg_.val_dataset->names()
                         : std::vector<std::string>{};
        auto out = (fs::path(cfg_.save_dir) /
                    ("train_batch" + std::to_string(step) + ".jpg")).string();
        render_train_batch(batch.imgs, batch.targets, names, out);
      }

      auto feats = model_->forward_train(imgs);
      // ProgLoss progress: linearly ramp 0 → 1 across all training steps.
      // (Yolo26Loss uses it; the default trait ignores the argument.)
      double progress = (total_steps > 1)
                            ? (double)gstep / (double)(total_steps - 1)
                            : 1.0;
      auto lo = Traits::compute(loss, feats, tgt, model_->stride,
                                cfg_.imgsz, progress);

      optim.zero_grad();
      lo.total.backward();
      // Average gradients across ranks (no-op single-process).
      all_reduce_grads(ddp, *model_);
      // Gradient clip (Ultralytics default 10.0)
      torch::nn::utils::clip_grad_norm_(model_->parameters(), /*max_norm=*/10.0);
      optim.step();

      // EMA
      ema_step++;
      double decay = cfg_.ema_decay *
                     (1.0 - std::exp(-(double)ema_step / cfg_.ema_warmup));
      ema_update(decay);

      sum_box   += lo.box.template item<double>();
      sum_cls   += lo.cls.template item<double>();
      sum_dfl   += lo.dfl.template item<double>();
      sum_total += lo.total.template item<double>();

      if (is_rank0(ddp) && gstep % cfg_.log_every == 0) {
        std::cout << "[trainer] e=" << epoch << " s=" << step
                  << " lr=" << lr_main
                  << " box=" << lo.box.template item<double>()
                  << " cls=" << lo.cls.template item<double>()
                  << " dfl=" << lo.dfl.template item<double>()
                  << " total=" << lo.total.template item<double>() << "\n";
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    if (is_rank0(ddp)) {
      std::cout << "[trainer] epoch " << epoch
                << " avg total=" << (sum_total / steps)
                << " (box=" << (sum_box / steps)
                << ", cls=" << (sum_cls / steps)
                << ", dfl=" << (sum_dfl / steps) << ")"
                << " in " << sec << "s\n";
    }

    // Optional validation + best-checkpoint tracking + early stopping.
    // Only rank 0 runs val (all ranks have synchronised weights via the
    // grad all-reduce path; running on every rank would just duplicate IO).
    double cur_map_50 = -1.0, cur_map_50_95 = -1.0;
    if (is_rank0(ddp) && cfg_.val_every > 0 && cfg_.val_dataset &&
        (epoch + 1) % cfg_.val_every == 0) {
      auto res = validate(ema_, *cfg_.val_dataset, device_);
      cur_map_50    = res.map_50;
      cur_map_50_95 = res.map_50_95;
      std::cout << "[trainer] val: mAP@0.5=" << res.map_50
                << " mAP@0.5:0.95=" << res.map_50_95 << "\n";
      if (res.map_50_95 > best_map) {
        best_map = res.map_50_95;
        best_epoch = epoch;
        epochs_since_best = 0;
        save_ckpt("best.pt");
        std::cout << "[trainer] new best mAP@0.5:0.95=" << best_map
                  << " (epoch " << epoch << ")\n";
      } else {
        ++epochs_since_best;
      }
      model_->train();
    }

    if (is_rank0(ddp)) {
      // results.csv row (rank 0 only).
      csv << epoch << "," << sec << ","
          << (sum_box / steps) << "," << (sum_cls / steps) << ","
          << (sum_dfl / steps) << ","
          << cur_map_50 << "," << cur_map_50_95 << ","
          << static_cast<torch::optim::SGDOptions&>(
                optim.param_groups()[0].options()).lr()
          << "\n";
      csv.flush();
    }

    // Early stopping check.
    if (cfg_.patience > 0 && epochs_since_best >= cfg_.patience) {
      std::cout << "[trainer] early stop: no improvement in "
                << cfg_.patience << " epochs (best mAP@0.5:0.95="
                << best_map << " at epoch " << best_epoch << ")\n";
      break;
    }
  }
  if (is_rank0(ddp)) csv.close();

  // Final saves + confusion matrix on rank 0 only.
  if (is_rank0(ddp)) {
    save_ckpt("last.pt");
    if (best_map >= 0.0) {
      std::cout << "[trainer] best.pt at mAP@0.5:0.95=" << best_map << "\n";
    } else {
      std::cout << "[trainer] no val dataset → best.pt not saved (only last.pt)\n";
    }
    if (cfg_.val_dataset) {
      auto vo = validate_with_records(ema_, *cfg_.val_dataset, device_);
      const auto& names = cfg_.val_dataset->names();
      int nc = cfg_.val_dataset->num_classes();

      // 1) Confusion matrix.
      auto cm = metrics::confusion_matrix(vo.dets, vo.gts, nc,
                                           /*conf=*/0.25, /*iou=*/0.45);
      auto cm_png = (fs::path(cfg_.save_dir) / "confusion_matrix.png").string();
      render_confusion_matrix(cm, names, cm_png);
      std::cout << "[trainer] saved → " << cm_png << "\n";

      // 2) PR / F1 / P / R curves.
      auto cd = metrics::compute_curves(vo.dets, vo.gts, nc);
      auto sd = cfg_.save_dir;
      render_curve_png(cd.pr_p, cd.pr_r, names, cd.n_gt_per_class,
                       "Recall", "Precision",
                       "Precision-Recall Curve",
                       (fs::path(sd) / "BoxPR_curve.png").string());
      render_curve_png(cd.f1, cd.px, names, cd.n_gt_per_class,
                       "Confidence", "F1",
                       "F1-Confidence Curve",
                       (fs::path(sd) / "BoxF1_curve.png").string());
      render_curve_png(cd.p, cd.px, names, cd.n_gt_per_class,
                       "Confidence", "Precision",
                       "Precision-Confidence Curve",
                       (fs::path(sd) / "BoxP_curve.png").string());
      render_curve_png(cd.r, cd.px, names, cd.n_gt_per_class,
                       "Confidence", "Recall",
                       "Recall-Confidence Curve",
                       (fs::path(sd) / "BoxR_curve.png").string());
      std::cout << "[trainer] saved → BoxPR_curve.png, BoxF1_curve.png,"
                   " BoxP_curve.png, BoxR_curve.png\n";

      // 3) labels.jpg — per-class GT count histogram.
      render_labels_histogram(vo.gts, names,
                              (fs::path(sd) / "labels.jpg").string());

      // 4) results.png — training curves from results.csv we just wrote.
      render_results_png((fs::path(sd) / "results.csv").string(),
                         (fs::path(sd) / "results.png").string());
    }
  }

  finalize_ddp(ddp);
}

// Explicit instantiations.
template class TrainerT<models::Yolo8Detect>;
template class TrainerT<models::Yolo5Detect>;
template class TrainerT<models::Yolo11Detect>;
template class TrainerT<models::Yolo12Detect>;
template class TrainerT<models::Yolo13Detect>;
template class TrainerT<models::Yolo26Detect>;

}  // namespace yolocpp::engine
