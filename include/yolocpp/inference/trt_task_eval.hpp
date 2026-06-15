#pragma once
//
// TRT-backed model adapters for non-detect task validation / benchmarking.
//
// The templated task validators (`validate_segment_t` / `validate_pose_t` /
// `validate_obb_t`) only ever call `model->to(device)`, `model->eval()`, and
// `model->forward_eval(x)` (verified). These thin adapters present exactly that
// interface but run the forward on a deserialized TensorRT engine instead of
// the LibTorch model — so the *entire* mask-/OKS-/rotated-AP metric is reused
// unchanged for the TRT formats, with the engine's multi-output tensors mapped
// back into the (pred, …) tuple each task expects.
//
// nvinfer lives only in `trt_task_eval.cpp`: the runner here is a plain
// `std::function`, so the task TUs that instantiate the validators stay
// TensorRT-free.
//

#include <torch/torch.h>

#include <functional>
#include <map>
#include <string>
#include <tuple>

namespace yolocpp::inference {

// Runs a TRT engine forward on a [1,3,H,W] input and returns every engine
// output tensor keyed by its name (owning CUDA tensors). Built by
// `make_trt_multi_forward`.
using TrtMultiForward =
    std::function<std::map<std::string, torch::Tensor>(torch::Tensor)>;

// Load `engine_path` and return a runner. `imgsz` is updated in place to the
// engine's static spatial size when it declares one (e.g. obb builds at 1024).
TrtMultiForward make_trt_multi_forward(const std::string& engine_path,
                                       int& imgsz /*in/out*/);

// ── Per-task model adapters ────────────────────────────────────────────────
// `operator->` returns self so `model->forward_eval(...)` resolves; `to`/`eval`
// are no-ops (the engine is already on-device and inference-only).

struct TrtSegModel {
  TrtMultiForward fwd;
  TrtSegModel*       operator->()       { return this; }
  const TrtSegModel* operator->() const { return this; }
  void to(torch::Device) {}
  void eval() {}
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward_eval(torch::Tensor x) {
    auto o = fwd(std::move(x));
    return {o.at("output"), o.at("coefs"), o.at("protos")};
  }
};

struct TrtPoseModel {
  TrtMultiForward fwd;
  TrtPoseModel*       operator->()       { return this; }
  const TrtPoseModel* operator->() const { return this; }
  void to(torch::Device) {}
  void eval() {}
  std::tuple<torch::Tensor, torch::Tensor>
  forward_eval(torch::Tensor x) {
    auto o = fwd(std::move(x));
    return {o.at("output"), o.at("keypoints")};
  }
};

struct TrtOBBModel {
  TrtMultiForward fwd;
  TrtOBBModel*       operator->()       { return this; }
  const TrtOBBModel* operator->() const { return this; }
  void to(torch::Device) {}
  void eval() {}
  std::tuple<torch::Tensor, torch::Tensor>
  forward_eval(torch::Tensor x) {
    auto o = fwd(std::move(x));
    return {o.at("output"), o.at("angle")};
  }
};

}  // namespace yolocpp::inference
