#include "yolocpp/tasks/classify_train.hpp"

#include "yolocpp/inference/trt_task_eval.hpp"  // TrtClassifyModel (TRT val)
#include "yolocpp/serialization/pt_save.hpp"     // save_module_state_dict

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/optim.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "yolocpp/inference/letterbox.hpp"

namespace fs = std::filesystem;

namespace yolocpp::tasks {

ClassifyDataset::ClassifyDataset(std::string root, std::string split,
                                 int imgsz, bool augment)
    : imgsz_(imgsz), augment_(augment) {
  fs::path split_dir = fs::path(root) / split;
  if (!fs::exists(split_dir))
    throw std::runtime_error("classify split missing: " + split_dir.string());

  // Each subdirectory of split_dir is a class.
  for (auto& e : fs::directory_iterator(split_dir)) {
    if (e.is_directory()) class_names_.push_back(e.path().filename().string());
  }
  std::sort(class_names_.begin(), class_names_.end());

  for (size_t ci = 0; ci < class_names_.size(); ++ci) {
    fs::path cdir = split_dir / class_names_[ci];
    for (auto& e : fs::directory_iterator(cdir)) {
      if (!e.is_regular_file()) continue;
      auto ext = e.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        paths_.push_back(e.path().string());
        labels_.push_back((int)ci);
      }
    }
  }
  if (paths_.empty())
    throw std::runtime_error("no images under " + split_dir.string());
}

ClassifyExample ClassifyDataset::get(std::size_t idx, uint64_t seed) const {
  ClassifyExample ex;
  ex.path = paths_[idx];
  ex.label = labels_[idx];
  cv::Mat img = cv::imread(ex.path, cv::IMREAD_COLOR);
  if (img.empty()) throw std::runtime_error("could not load: " + ex.path);

  // Resize-shortest + center/random crop. Use INTER_AREA when downsampling to
  // match torchvision's antialiased BILINEAR (and the inference predictor's
  // run_classify) — plain INTER_LINEAR aliases and shifts the train-time pixel
  // distribution away from what inference/val sees.
  int short_side = std::min(img.rows, img.cols);
  double scale = (double)imgsz_ / short_side;
  cv::Mat resized;
  int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::resize(img, resized, {}, scale, scale, interp);

  std::mt19937 rng(seed ? seed : (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
  int x0, y0;
  if (augment_) {
    std::uniform_int_distribution<int> ux(0, std::max(0, resized.cols - imgsz_));
    std::uniform_int_distribution<int> uy(0, std::max(0, resized.rows - imgsz_));
    x0 = ux(rng); y0 = uy(rng);
  } else {
    x0 = (resized.cols - imgsz_) / 2;
    y0 = (resized.rows - imgsz_) / 2;
  }
  cv::Mat cropped = resized(cv::Rect(x0, y0, imgsz_, imgsz_)).clone();

  if (augment_) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    if (u(rng) < 0.5f) cv::flip(cropped, cropped, /*flipCode=*/1);
  }

  ex.img = inference::image_to_tensor(cropped);
  return ex;
}

ClassifyDataset::Batch ClassifyDataset::sample_batch(std::size_t bsz,
                                                      std::mt19937& rng) const {
  std::uniform_int_distribution<size_t> u(0, paths_.size() - 1);
  std::vector<torch::Tensor> imgs;
  std::vector<int64_t>       labels;
  for (size_t i = 0; i < bsz; ++i) {
    auto ex = get(u(rng), (uint64_t)rng() << 32 | i);
    imgs.push_back(ex.img);
    labels.push_back(ex.label);
  }
  Batch b;
  b.imgs   = torch::stack(imgs, 0);
  b.labels = torch::tensor(labels, torch::kLong);
  return b;
}

static torch::Device pick_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}

template <typename M>
void train_classify_t(M model,
                      const ClassifyDataset& train,
                      const ClassifyDataset* val,
                      ClassifyTrainConfig cfg) {
  fs::create_directories(cfg.save_dir);
  auto device = pick_device(cfg.device);
  model->to(device);
  model->train();

  // Single param group with weight decay (we keep this simple for classify).
  std::vector<at::Tensor> params;
  for (auto& p : model->parameters())
    if (p.requires_grad()) params.push_back(p);

  auto opt_options = torch::optim::SGDOptions(cfg.lr0)
                          .momentum(cfg.momentum)
                          .weight_decay(cfg.weight_decay)
                          .nesterov(true);
  torch::optim::SGD optim(params, opt_options);

  std::mt19937 rng(0x9E3779B9u);
  size_t n     = train.size();
  int    steps = std::max<int>(1, (int)(n / cfg.batch_size));
  int    total_steps  = steps * cfg.epochs;
  int    warmup_steps = std::max(50, steps * cfg.warmup_epochs);
  std::cout << "[cls-train] " << n << " imgs, " << steps << " steps/epoch, "
            << "device=" << device << "\n";

  for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
    auto t0 = std::chrono::steady_clock::now();
    double running_loss = 0;
    for (int step = 0; step < steps; ++step) {
      int gstep = epoch * steps + step;
      double scale;
      if (gstep < warmup_steps)
        scale = (double)(gstep + 1) / (double)warmup_steps;
      else {
        double t = (double)(gstep - warmup_steps) /
                   std::max(1.0, (double)(total_steps - warmup_steps));
        scale = cfg.lrf + 0.5 * (1.0 - cfg.lrf) * (1.0 + std::cos(M_PI * t));
      }
      auto& gs = optim.param_groups();
      static_cast<torch::optim::SGDOptions&>(gs[0].options()).lr(cfg.lr0 * scale);

      auto b = train.sample_batch(cfg.batch_size, rng);
      auto x = b.imgs.to(device);
      auto y = b.labels.to(device);
      auto logits = model->forward(x);
      auto loss   = torch::nn::functional::cross_entropy(logits, y);

      optim.zero_grad();
      loss.backward();
      torch::nn::utils::clip_grad_norm_(model->parameters(), 10.0);
      optim.step();

      running_loss += loss.template item<double>();
      if (gstep % cfg.log_every == 0) {
        std::cout << "[cls-train] e=" << epoch << " s=" << step
                  << " lr=" << cfg.lr0 * scale
                  << " loss=" << loss.template item<double>() << "\n";
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[cls-train] epoch " << epoch << " avg_loss="
              << (running_loss / steps)
              << " in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    if (val && cfg.val_every > 0 && (epoch + 1) % cfg.val_every == 0) {
      auto vr = validate_classify_t(model, *val, device);
      std::cout << "[cls-train] val top1=" << vr.top1_acc
                << " top5=" << vr.top5_acc << "\n";
      model->train();
    }
  }

  auto ckpt = fs::path(cfg.save_dir) / "last.pt";
  serialization::save_module_state_dict(*model, ckpt.string());
  std::cout << "[cls-train] saved → " << ckpt << "\n";
}

template <typename M>
ClassifyValResult validate_classify_t(M& model,
                                       const ClassifyDataset& dataset,
                                       torch::Device device) {
  model->to(device);
  model->eval();
  int total = 0, top1 = 0, top5 = 0;
  for (size_t i = 0; i < dataset.size(); ++i) {
    auto ex = dataset.get(i, /*seed=*/i + 1);
    auto x = ex.img.unsqueeze(0).to(device);
    torch::Tensor logits;
    {
      torch::NoGradGuard ng;
      logits = model->forward(x);
    }
    auto probs = logits.softmax(1).cpu();
    int k = std::min(5, (int)probs.size(1));
    auto topk = probs.topk(k, 1, true, true);
    auto idx  = std::get<1>(topk).accessor<int64_t, 2>();
    if ((int)idx[0][0] == ex.label) ++top1;
    for (int j = 0; j < k; ++j)
      if ((int)idx[0][j] == ex.label) { ++top5; break; }
    ++total;
  }
  ClassifyValResult r;
  r.n_total  = total;
  r.top1_acc = total ? (double)top1 / total : 0.0;
  r.top5_acc = total ? (double)top5 / total : 0.0;
  return r;
}

// Explicit instantiations for both Yolo8Classify and Yolo11Classify.
template void train_classify_t<models::Yolo8Classify>(
    models::Yolo8Classify, const ClassifyDataset&,
    const ClassifyDataset*, ClassifyTrainConfig);
template void train_classify_t<models::Yolo11Classify>(
    models::Yolo11Classify, const ClassifyDataset&,
    const ClassifyDataset*, ClassifyTrainConfig);
template ClassifyValResult validate_classify_t<models::Yolo8Classify>(
    models::Yolo8Classify&, const ClassifyDataset&, torch::Device);
template ClassifyValResult validate_classify_t<models::Yolo11Classify>(
    models::Yolo11Classify&, const ClassifyDataset&, torch::Device);
template void train_classify_t<models::Yolo12Classify>(
    models::Yolo12Classify, const ClassifyDataset&, const ClassifyDataset*,
    ClassifyTrainConfig);
template ClassifyValResult validate_classify_t<models::Yolo12Classify>(
    models::Yolo12Classify&, const ClassifyDataset&, torch::Device);
// TRT-backed validation (per-format benchmark top-1) reuses the same metric.
template ClassifyValResult validate_classify_t<inference::TrtClassifyModel>(
    inference::TrtClassifyModel&, const ClassifyDataset&, torch::Device);

}  // namespace yolocpp::tasks
