#include "yolocpp/engine/trainer.hpp"

#include <ATen/autocast_mode.h>
#include <torch/optim.h>
#include <torch/optim/adamw.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "yolocpp/engine/ddp.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/losses/yolo1_loss.hpp"
#include "yolocpp/losses/yolo26_loss.hpp"
#include "yolocpp/losses/yolo2_loss.hpp"
#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::engine {

namespace fs = std::filesystem;

// Pick a fresh "<base><N>" directory: returns base if it doesn't exist,
// else base2, base3, … — matches the upstream `runs/<mode>/<exp>` convention.
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

// BatchPrefetcher — N background threads each call
// `dataset.sample_batch` into a shared bounded queue; the main
// training loop pulls from the queue via `next()`. This pipelines
// data prep (decode + mosaic + perspective + letterbox + tensor
// conversion) ahead of the GPU step, hiding the CPU latency that
// was the single biggest 0.93.0 perf gap vs Ultralytics' Python
// DataLoader-with-workers (~2× per-epoch wall time on yolo26x +
// mosaic). Workers each own their own seeded `std::mt19937` so
// reproducibility holds within each worker; the across-worker
// batch ordering is non-deterministic (acceptable: SGD/AdamW
// gradient estimates are over-batch averages — the per-step batch
// composition is what matters, not its global ordering).
class BatchPrefetcher {
 public:
  BatchPrefetcher(const datasets::YoloDataset& ds, int batch_size,
                  int num_workers, uint64_t seed)
      : ds_(ds), batch_size_(batch_size),
        num_workers_(std::max(1, num_workers)),
        max_q_((std::size_t)(2 * std::max(1, num_workers))) {
    workers_.reserve((std::size_t)num_workers_);
    const uint64_t base = seed != 0 ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < num_workers_; ++i) {
      // Distinct, well-separated per-worker seeds (golden-ratio
      // hash splat). Each worker's sample_batch produces a
      // deterministic stream given its own seed.
      const uint64_t ws = base + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
      workers_.emplace_back([this, ws] { worker_loop_(ws); });
    }
  }

  ~BatchPrefetcher() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_space_.notify_all();
    cv_data_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
  }

  // Blocking pop. Returns the next batch; the caller owns it.
  datasets::YoloDataset::Batch next() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_data_.wait(lk, [this] { return !q_.empty() || stop_; });
    auto b = std::move(q_.front());
    q_.pop_front();
    cv_space_.notify_one();
    return b;
  }

 private:
  void worker_loop_(uint64_t seed) {
    std::mt19937 rng(seed);
    while (true) {
      // Sample a batch off-lock — this is the heavy CPU work
      // (image decode + mosaic + perspective + letterbox) we want
      // overlapping with the main thread's GPU step.
      auto batch = ds_.sample_batch((std::size_t)batch_size_, rng);
      std::unique_lock<std::mutex> lk(mu_);
      cv_space_.wait(lk, [this] { return q_.size() < max_q_ || stop_; });
      if (stop_) return;
      q_.push_back(std::move(batch));
      cv_data_.notify_one();
    }
  }

  const datasets::YoloDataset&            ds_;
  int                                     batch_size_;
  int                                     num_workers_;
  std::size_t                             max_q_;
  std::deque<datasets::YoloDataset::Batch> q_;
  std::mutex                              mu_;
  std::condition_variable                 cv_data_, cv_space_;
  bool                                    stop_ = false;
  std::vector<std::thread>                workers_;
};

// Build the per-group optimizer. Three groups (decay-conv / bias / bn-weight)
// for both SGD and AdamW — same partition, different per-group options.
//
// AdamW defaults match ultralytics' auto-pick when fine-tuning from a
// pretrained .pt at batch=16: lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
// weight_decay split = (cfg.weight_decay, 0, 0). The adaptive
// per-parameter scaling protects the v26 cls-head prior bias from
// the same overshoot that SGD+lr=0.01 produces — which is why the v26
// LR-override in `version_registry.cpp` only fires for SGD.
std::unique_ptr<torch::optim::Optimizer> make_optimizer(
    const std::string& kind_in,
    const std::vector<at::Tensor>& decay,
    const std::vector<at::Tensor>& bias,
    const std::vector<at::Tensor>& bn,
    double lr0, double momentum, double weight_decay) {
  auto kind = kind_in;  // already-resolved by resolve_optimizer_kind
  std::vector<torch::optim::OptimizerParamGroup> groups;
  if (kind == "adamw") {
    auto add = [&](const std::vector<at::Tensor>& ps, double wd) {
      auto g = torch::optim::OptimizerParamGroup(ps);
      auto opt = std::make_unique<torch::optim::AdamWOptions>(lr0);
      opt->betas({0.9, 0.999}).eps(1e-8).weight_decay(wd);
      g.set_options(std::move(opt));
      groups.push_back(std::move(g));
    };
    add(decay, weight_decay);
    add(bias,  0.0);
    add(bn,    0.0);
    return std::make_unique<torch::optim::AdamW>(
        std::move(groups), torch::optim::AdamWOptions(lr0).weight_decay(weight_decay));
  }
  // SGD + Nesterov (default for batch >= 64 or explicit "sgd").
  auto add = [&](const std::vector<at::Tensor>& ps, double wd) {
    auto g = torch::optim::OptimizerParamGroup(ps);
    auto opt = std::make_unique<torch::optim::SGDOptions>(lr0);
    opt->momentum(momentum).weight_decay(wd).nesterov(true);
    g.set_options(std::move(opt));
    groups.push_back(std::move(g));
  };
  add(decay, weight_decay);
  add(bias,  0.0);
  add(bn,    0.0);
  return std::make_unique<torch::optim::SGD>(
      std::move(groups), torch::optim::SGDOptions(lr0).momentum(momentum).nesterov(true));
}

// Build three parameter groups (decay-conv, no-decay-bias-bn, BN-weight-no-decay)
// matching the upstream SGD setup.
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
// Walk the training set once and return every GT (w, h) in pixel
// units (after the dataset's letterbox to imgsz). Sampling cap of
// ~10000 boxes is plenty for stable K-means.
inline std::vector<std::pair<float, float>>
collect_train_gt_whs(const datasets::YoloDataset& train) {
  std::vector<std::pair<float, float>> gt_whs;
  gt_whs.reserve(10000);
  const std::size_t cap = 10000;
  for (std::size_t i = 0; i < train.size() && gt_whs.size() < cap; ++i) {
    auto ex = train.get(i);
    if (!ex.targets.defined() || ex.targets.size(0) == 0) continue;
    auto t = ex.targets.to(torch::kCPU).to(torch::kFloat32);
    auto acc = t.accessor<float, 2>();
    for (int j = 0; j < t.size(0); ++j) {
      float w = acc[j][3], h = acc[j][4];
      if (w > 1.0f && h > 1.0f) gt_whs.emplace_back(w, h);
      if (gt_whs.size() >= cap) break;
    }
  }
  return gt_whs;
}

// Post-construction hook: gives anchor-based losses (v4/v7) a chance
// to recluster anchors from the training-set GT distribution AND
// sync the matching model anchor buffers so eval/decoding sees the
// same anchors as training. Default no-op — DFL-headed losses
// (v8/v11/v12/v13/v10/v26) have no static anchors. Specialized below
// for v7 (and shareable for v4 via the model holder's `anchors` field
// if added later).
template <typename M, typename LossT>
void loss_after_init(M& /*model*/, LossT& /*loss*/,
                     const datasets::YoloDataset& /*train*/) {}

template <>
inline void loss_after_init<models::Yolo7, losses::V7DetectionLoss>(
    models::Yolo7& model, losses::V7DetectionLoss& loss,
    const datasets::YoloDataset& train) {
  if (!loss.config().autoanchor) return;
  auto gt_whs = collect_train_gt_whs(train);
  loss.recluster_anchors(gt_whs);
  // Sync the model's IDetect anchor buffers so eval/decoding uses
  // the reclustered table — otherwise training optimizes against
  // the new anchors but inference decodes with the COCO ones.
  const auto& cfg = loss.config();
  const int nl = (int)cfg.strides.size();
  const int na = cfg.na;
  // Find IDetect in the model's ModuleList — it's the last entry.
  auto& yaml = models::yolo7_yaml_for(model->scale);
  auto* d = model->model[yaml.size() - 1]->as<models::IDetectImpl>();
  if (!d || !d->anchor_grid.defined() || !d->anchors.defined()) return;
  torch::NoGradGuard ng;
  for (int li = 0; li < nl; ++li) {
    for (int a = 0; a < na; ++a) {
      d->anchor_grid[li][0][a][0][0][0] = cfg.anchors[li][a].first;
      d->anchor_grid[li][0][a][0][0][1] = cfg.anchors[li][a].second;
      d->anchors[li][a][0] = cfg.anchors[li][a].first  / (float)cfg.strides[li];
      d->anchors[li][a][1] = cfg.anchors[li][a].second / (float)cfg.strides[li];
    }
  }
}

template <typename M>
struct LossTraits {
  using LossT   = losses::V8DetectionLoss;
  using OutputT = losses::LossOutput;
  static LossT make(const M& model) {
    losses::LossConfig c; c.nc = model->nc; c.reg_max = 16;
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
  static LossT make(const models::Yolo26Detect& model) {
    losses::Yolo26LossConfig c; c.nc = model->nc;
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

// Specialised for Yolo4 → V7DetectionLoss with v4-specific anchors +
// scale_xy bias-fix [1.2, 1.1, 1.05] and exp() wh decode (Darknet-style).
template <>
struct LossTraits<models::Yolo4> {
  using LossT   = losses::V7DetectionLoss;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo4& model) {
    losses::V7LossConfig c;
    c.nc = model->nc; c.na = 3;
    // v4 anchors at imgsz=608 (yolov4.cfg). P3 → P4 → P5 order.
    c.anchors = {
      {{12.f, 16.f}, {19.f, 36.f}, {40.f, 28.f}},        // P3
      {{36.f, 75.f}, {76.f, 55.f}, {72.f, 146.f}},       // P4
      {{142.f, 110.f}, {192.f, 243.f}, {459.f, 401.f}},  // P5
    };
    c.strides  = {8, 16, 32};
    c.scale_xy = {1.2f, 1.1f, 1.05f};
    c.wh_sigmoid = false;   // v4 uses exp() wh (Darknet form)
    c.balance = {4.0f, 1.0f, 0.4f};
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

// Specialised for Yolo10 → Yolo10LossAdapter (auto-routes between
// V8DetectionLoss for single-head and V10DualLoss for dual-head training
// based on `forward_train`'s output count: 3 = one2one only, 6 =
// {one2many P3..P5, one2one P3..P5} for paper §3.1 consistent assignment).
template <>
struct LossTraits<models::Yolo10> {
  using LossT   = losses::Yolo10LossAdapter;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo10& model) {
    losses::LossConfig c; c.nc = model->nc; c.reg_max = 16;
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

// Specialised for Yolo7 → V7DetectionLoss with WongKinYiu's anchors and
// uniform scale_xy=2.0 + (sigmoid*2)^2 wh decode.
//
// Picks 3-level P5 anchors for base/tiny/x, 4-level P6 anchors for
// w6/e6/d6/e6e — all four P6 variants share the same yolov7-w6.yaml
// anchor table. Without this, the P6 head's 4 feature maps were fed
// into a 3-level loss config and the trainer segfaulted on the first
// batch.
template <>
struct LossTraits<models::Yolo7> {
  using LossT   = losses::V7DetectionLoss;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo7& model) {
    using S = models::Yolo7Scale;
    const bool p6 = (model->scale == S::W6 || model->scale == S::E6 ||
                     model->scale == S::D6 || model->scale == S::E6e);
    losses::V7LossConfig c;
    c.nc = model->nc; c.na = 3;
    if (p6) {
      // yolov7-w6.yaml / -e6.yaml / -d6.yaml / -e6e.yaml all use the
      // same anchor table at imgsz=1280. P3 → P4 → P5 → P6.
      c.anchors = {
        {{ 19.f,  27.f}, { 44.f,  40.f}, { 38.f,  94.f}},    // P3 / 8
        {{ 96.f,  68.f}, { 86.f, 152.f}, {180.f, 137.f}},    // P4 / 16
        {{140.f, 301.f}, {303.f, 264.f}, {238.f, 542.f}},    // P5 / 32
        {{436.f, 615.f}, {739.f, 380.f}, {925.f, 792.f}},    // P6 / 64
      };
      c.strides  = {8, 16, 32, 64};
      c.scale_xy = {2.0f, 2.0f, 2.0f, 2.0f};
      c.balance  = {4.0f, 1.0f, 0.4f, 0.1f};
    } else {
      // v7-base/tiny/x anchors at imgsz=640 (yolov7.yaml).
      c.anchors = {
        {{ 12.f,  16.f}, { 19.f,  36.f}, { 40.f,  28.f}},    // P3
        {{ 36.f,  75.f}, { 76.f,  55.f}, { 72.f, 146.f}},    // P4
        {{142.f, 110.f}, {192.f, 243.f}, {459.f, 401.f}},    // P5
      };
      c.strides  = {8, 16, 32};
      c.scale_xy = {2.0f, 2.0f, 2.0f};
      c.balance  = {4.0f, 1.0f, 0.4f};
    }
    c.wh_sigmoid = true;
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

// Specialised for Yolo6 → V6DetectionLoss (VFL + SIoU + TAL).
// Targets the DFL-headed v6 variants (m/l/m6/l6 — reg_max=16). For
// n/s/n6/s6 the head's `reg_preds_dist` branch (training-time KD
// target upstream) carries the 68-ch DFL distribution and is what
// `forward_train_per_scale_n` returns, so the loss path works for
// those variants too — just skip-to-deploy converts the model into
// the direct-4-ch `reg_preds` branch.
template <>
struct LossTraits<models::Yolo6> {
  using LossT   = losses::V6DetectionLoss;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo6& model) {
    losses::V6LossConfig c; c.nc = model->nc; c.reg_max = 16;
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

// Specialised for Yolo1 → SSE loss (Redmon 2016, λ_coord=5, λ_noobj=0.5).
template <>
struct LossTraits<models::Yolo1> {
  using LossT   = losses::Yolo1Loss;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo1& model) {
    losses::Yolo1LossConfig c;
    c.S = model->S; c.B = model->B; c.nc = model->nc;
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

// Specialised for Yolo2 → region loss with k-means anchors.
template <>
struct LossTraits<models::Yolo2> {
  using LossT   = losses::Yolo2Loss;
  using OutputT = losses::LossOutput;
  static LossT make(const models::Yolo2& model) {
    losses::Yolo2LossConfig c;
    c.nc      = model->nc;
    c.anchors = model->anchors;
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

// Build an EMA-clone of `src`. Default: invoke `M(scale, nc)`. Specialized
// per-model when the `(scale, nc)` ctor would erase config the EMA needs
// to share with the live model — currently `Yolo6Impl::is_p6` (the
// P5/P6 head topology is not part of `Yolo6Scale`, so `Yolo6(scale, nc)`
// always builds a P5 even when the live model is P6, and the subsequent
// `ema_->named_parameters().copy_(...)` in this ctor fails with a shape
// mismatch on the first P6-only parameter — historically reported as
// "tensor a (256) must match tensor b (192) at dim 0").
template <typename M>
M make_ema_clone(const M& src) { return M(src->scale, src->nc); }

template <>
models::Yolo6 make_ema_clone<models::Yolo6>(const models::Yolo6& src) {
  return models::Yolo6(src->nc, src->scale, src->reg_max, src->is_p6);
}

// Yolo1 has no `scale` member (single-variant model).
template <>
models::Yolo1 make_ema_clone<models::Yolo1>(const models::Yolo1& src) {
  return models::Yolo1(src->nc, src->S, src->B);
}

// Yolo2's `scale` is of type Yolo2Scale, not a string letter — the
// holder ctor takes (scale, nc, anchors).
template <>
models::Yolo2 make_ema_clone<models::Yolo2>(const models::Yolo2& src) {
  return models::Yolo2(src->scale, src->nc, src->anchors);
}

template <typename M>
TrainerT<M>::TrainerT(M model, datasets::YoloDataset train, TrainConfig cfg)
    : model_(std::move(model)),
      ema_(make_ema_clone(model_)),
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
  // CUDA perf globals — see CLAUDE.md "Training speed". cuDNN benchmark
  // picks fastest conv algo per shape; TF32 stays on for training (the
  // inverse of the v10-TRT-export quirk). Idempotent: safe to re-enter.
  if (torch::cuda::is_available()) {
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
  }

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

  // Dump the run's input args for reproducibility — `args.yaml` in the
  // upstream shape so existing tooling can consume it. The list of keys
  // and default values follows the upstream ≥ 8.3 default.yaml schema.
  if (is_rank0(ddp)) {
    std::ofstream y((fs::path(cfg_.save_dir) / "args.yaml").string());

    // Helper: was this key supplied on the CLI? If so, use that value;
    // otherwise emit the upstream default.
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
    y << "# (matches the upstream ≥ 8.3 default.yaml field list)\n";

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
    // Real value — BatchPrefetcher uses cfg_.workers background
    // threads (0 = synchronous fallback).
    emit("workers",     std::to_string(cfg_.workers));
    y << "project:\n";
    y << "name:\n";
    emit("exist_ok",    "false");
    emit("pretrained",  kv_lookup("model").empty() ? "true" : "true");
    emit("optimizer",   "auto");
    emit("verbose",     "true");
    emit("seed",        std::to_string(cfg_.seed));
    emit("deterministic","true");
    emit("single_cls",  "false");
    emit("rect",        "false");
    emit("cos_lr",      "false");
    emit("close_mosaic","10");
    emit("resume",      "false");
    // AMP is on (bf16 autocast) when training on CUDA; off on CPU.
    emit("amp",         torch::cuda::is_available() ? "true" : "false");
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
  // channels_last on CUDA — extra 10-20% on Tensor Cores for the
  // conv-heavy YOLO backbones. LibTorch C++ has no Module::to(memformat),
  // so we walk 4D parameters + buffers ourselves. 1D tensors (BN stats,
  // biases) are layout-invariant and skipped. Input batches are
  // converted at sample time below to keep the format consistent.
  if (device_.is_cuda()) {
    auto to_nhwc = [](torch::nn::Module& m) {
      for (auto& p : m.parameters()) {
        if (p.dim() == 4) p.set_data(p.contiguous(torch::MemoryFormat::ChannelsLast));
      }
      for (auto& b : m.buffers()) {
        if (b.dim() == 4) b.set_data(b.contiguous(torch::MemoryFormat::ChannelsLast));
      }
    };
    to_nhwc(*model_);
    to_nhwc(*ema_);
  }
  // Sync rank-0 weights to all ranks before the first step.
  broadcast_module(ddp, *model_);
  broadcast_module(ddp, *ema_);
  model_->train();

  auto pg = split_params(*model_);
  // Resolve optimizer choice. "auto" → AdamW for batch < 64 (the
  // short-fine-tune regime), SGD otherwise. The factory wraps either
  // in std::unique_ptr<torch::optim::Optimizer>; the rest of the loop
  // talks to the polymorphic base only.
  const std::string opt_kind = resolve_optimizer(cfg_.optimizer, cfg_.batch_size);
  // When AdamW is auto-picked from the SGD-natural lr0 default (0.01),
  // scale lr0 by 0.1 to land at AdamW's natural ~1e-3 range (matches
  // ultralytics' auto-lr behaviour: 0.01 → 0.001111 ≈ /9 when auto
  // switches SGD → AdamW). Skipped when the user explicitly chose
  // "adamw" with a non-default lr0 — we assume they meant it.
  if (opt_kind == "adamw" && cfg_.optimizer == "auto" &&
      std::abs(cfg_.lr0 - 0.01) < 1e-9) {
    if (is_rank0(ddp)) {
      std::cout << "[trainer] auto-lr: optimizer=adamw → lr0 0.01 → 1e-3 "
                   "(matches upstream auto-lr). Pass --lr0 to override.\n";
    }
    cfg_.lr0 = 1e-3;
    if (std::abs(cfg_.warmup_bias_lr - 0.1) < 1e-9) cfg_.warmup_bias_lr = cfg_.lr0;
  }
  if (is_rank0(ddp)) {
    std::cout << "[trainer] optimizer=" << opt_kind
              << " (requested=" << cfg_.optimizer
              << ") lr0=" << cfg_.lr0 << "\n";
  }
  auto optim_ptr = make_optimizer(opt_kind, pg.decay, pg.bias, pg.bn,
                                  cfg_.lr0, cfg_.momentum, cfg_.weight_decay);
  auto& optim = *optim_ptr;

  using Traits = LossTraits<M>;
  auto loss = Traits::make(model_);
  // Anchor-based losses recluster from the train GTs (autoanchor)
  // and re-sync the matching model anchor buffers. No-op for DFL-
  // based losses.
  loss_after_init<M, typename Traits::LossT>(model_, loss, train_);

  // Seed every RNG when --seed is non-zero. Order: torch's CPU + CUDA
  // generators (operator-level), then the trainer's batch-sampler RNG.
  // Per-example augmentation RNGs are derived inside `train_.get(...)`
  // from (epoch, idx) folded with this same seed (see sample_batch
  // below).
  if (cfg_.seed != 0) {
    torch::manual_seed(static_cast<int64_t>(cfg_.seed));
    if (torch::cuda::is_available()) {
      torch::cuda::manual_seed_all(static_cast<int64_t>(cfg_.seed));
    }
  }
  std::mt19937 rng(cfg_.seed != 0 ? cfg_.seed : 0x9E3779B9u);
  size_t       n      = train_.size();
  int          steps  = std::max<int>(1, (int)(n / cfg_.batch_size));
  int          total_steps   = steps * cfg_.epochs;
  // Warmup: target = steps × warmup_epochs (upstream default 3 epochs).
  // Floor at 100 steps for big datasets, but never exceed half the total —
  // otherwise on tiny datasets all of training stays in linear warmup and
  // cosine decay never kicks in.
  // Cap warmup at 10% of total steps (not 50%). The upstream
  // warmup_epochs=3 default assumes 100+ epochs of training; for the
  // short 2–10 epoch budgets common in fine-tuning, 50% of training
  // spent in warmup leaves the effective LR <10% of `lr0` for the
  // entire run, which is what caused v26 fine-tuning to look
  // "stuck" — convergence was fine but LR was wasted.
  int          warmup_target = std::max(100, steps * cfg_.warmup_epochs);
  int          warmup_cap    = std::max(100, total_steps / 10);
  int          warmup_steps  = std::min(warmup_target, warmup_cap);
  warmup_steps = std::max(1, warmup_steps);
  int          ema_step      = 0;

  std::cout << "[trainer] dataset=" << n << " imgs, "
            << steps << " steps/epoch, total=" << total_steps
            << ", device=" << device_
            << ", workers=" << cfg_.workers << "\n";

  // BatchPrefetcher pipelines mosaic + perspective + decode + letterbox
  // ahead of the GPU step. Constructed inside run() so its lifetime is
  // strictly scoped to training — destructor stops + joins workers
  // before validate() / save_ckpt() / curve rendering touch the dataset.
  // workers=0 falls back to the synchronous per-step sample_batch path.
  std::unique_ptr<BatchPrefetcher> prefetch;
  if (cfg_.workers > 0) {
    prefetch = std::make_unique<BatchPrefetcher>(
        train_, cfg_.batch_size, cfg_.workers, cfg_.seed);
  }

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

  // Open results.csv. Columns match the upstream Ultralytics layout
  // so downstream tooling (results.png renderer, external plotters)
  // can read either framework's output. P/R/F1 are sampled at the
  // best-F1 confidence threshold (max of the F1 curve averaged across
  // classes), then averaged across classes weighted by GT count.
  std::ofstream csv((fs::path(cfg_.save_dir) / "results.csv").string());
  csv << "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
         "metrics/precision(B),metrics/recall(B),metrics/F1(B),"
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
      gs[0].options().set_lr(lr_main);
      gs[1].options().set_lr(lr_bias);
      gs[2].options().set_lr(lr_main);

      auto batch = prefetch
          ? prefetch->next()
          : train_.sample_batch(cfg_.batch_size, rng);
      auto imgs  = batch.imgs.to(device_);
      auto tgt   = batch.targets.to(device_);
      if (device_.is_cuda()) {
        // Match the model's channels_last layout so the conv kernels
        // pick the NHWC Tensor-Core path.
        imgs = imgs.contiguous(torch::MemoryFormat::ChannelsLast);
      }

      // Emit train_batchN.jpg sanity grids for the first 3 batches of the
      // first epoch — upstream convention.
      if (is_rank0(ddp) && epoch == 0 && step < 3) {
        auto names = cfg_.val_dataset
                         ? cfg_.val_dataset->names()
                         : std::vector<std::string>{};
        auto out = (fs::path(cfg_.save_dir) /
                    ("train_batch" + std::to_string(step) + ".jpg")).string();
        render_train_batch(batch.imgs, batch.targets, names, out);
      }

      // AMP via bf16 autocast on CUDA. bf16 carries fp32's dynamic
      // range so no GradScaler is needed (the Blackwell path). The
      // autocast scope wraps forward + loss only; backward + step run
      // with the loss tensor implicitly upcast to fp32 by autocast.
      const bool use_amp = device_.is_cuda();
      if (use_amp) {
        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
      }
      auto feats = model_->forward_train(imgs);
      // ProgLoss progress: linearly ramp 0 → 1 across all training steps.
      // (Yolo26Loss uses it; the default trait ignores the argument.)
      double progress = (total_steps > 1)
                            ? (double)gstep / (double)(total_steps - 1)
                            : 1.0;
      auto lo = Traits::compute(loss, feats, tgt, model_->stride,
                                cfg_.imgsz, progress);
      if (use_amp) {
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::clear_cache();
      }

      optim.zero_grad();
      lo.total.backward();
      // Average gradients across ranks (no-op single-process).
      all_reduce_grads(ddp, *model_);
      // Gradient clip (upstream default 10.0)
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
    double cur_p = -1.0, cur_r = -1.0, cur_f1 = -1.0;
    if (is_rank0(ddp) && cfg_.val_every > 0 && cfg_.val_dataset &&
        (epoch + 1) % cfg_.val_every == 0) {
      // validate_with_records returns the full det+gt rows so we can
      // run compute_curves and extract scalar P/R/F1 at best-F1.
      auto vo = validate_with_records(ema_, *cfg_.val_dataset, device_);
      auto& res = vo.map;
      cur_map_50    = res.map_50;
      cur_map_50_95 = res.map_50_95;
      const int nc_ = cfg_.val_dataset->num_classes();
      auto cd = metrics::compute_curves(vo.dets, vo.gts, nc_);
      // Mean F1 across classes (weighted by GT count) at each
      // confidence threshold; argmax gives the best operating point.
      const int L = (int)cd.px.size();
      double best_mean_f1 = -1.0;
      int    best_i = 0;
      double total_gt = 0.0;
      for (int c = 0; c < nc_; ++c) total_gt += (double)cd.n_gt_per_class[c];
      if (total_gt > 0.0 && nc_ > 0) {
        for (int i = 0; i < L; ++i) {
          double s = 0.0;
          for (int c = 0; c < nc_; ++c)
            s += cd.f1[c][i] * (double)cd.n_gt_per_class[c];
          double m = s / total_gt;
          if (m > best_mean_f1) { best_mean_f1 = m; best_i = i; }
        }
        double sp = 0.0, sr = 0.0;
        for (int c = 0; c < nc_; ++c) {
          sp += cd.p[c][best_i] * (double)cd.n_gt_per_class[c];
          sr += cd.r[c][best_i] * (double)cd.n_gt_per_class[c];
        }
        cur_p  = sp / total_gt;
        cur_r  = sr / total_gt;
        cur_f1 = best_mean_f1;
      }
      std::cout << "[trainer] val: mAP@0.5=" << res.map_50
                << " mAP@0.5:0.95=" << res.map_50_95
                << " P=" << cur_p << " R=" << cur_r << " F1=" << cur_f1 << "\n";
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
          << cur_p << "," << cur_r << "," << cur_f1 << ","
          << cur_map_50 << "," << cur_map_50_95 << ","
          << optim.param_groups()[0].options().get_lr()
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

  // Stop prefetcher workers before the final validate + curve render
  // pass so they don't contend on CPU for image decode.
  prefetch.reset();

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
template class TrainerT<models::Yolo3>;
template class TrainerT<models::Yolo4>;
template class TrainerT<models::Yolo5Detect>;
template class TrainerT<models::Yolo6>;
template class TrainerT<models::Yolo7>;
template class TrainerT<models::Yolo9>;
template class TrainerT<models::Yolo10>;
template class TrainerT<models::Yolo11Detect>;
template class TrainerT<models::Yolo12Detect>;
template class TrainerT<models::Yolo13Detect>;
template class TrainerT<models::Yolo26Detect>;
template class TrainerT<models::Yolo1>;
template class TrainerT<models::Yolo2>;

}  // namespace yolocpp::engine
