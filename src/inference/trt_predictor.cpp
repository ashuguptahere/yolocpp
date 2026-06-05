#include "yolocpp/inference/trt_predictor.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/core/profile.hpp"

namespace yolocpp::inference {

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
};

#define CUDA_OK(call)                                                   \
  do {                                                                  \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess)                                              \
      throw std::runtime_error(std::string("cuda: ") +                  \
                               cudaGetErrorString(_e));                 \
  } while (0)

}  // anonymous namespace

struct TrtPredictor::Impl {
  TrtLogger                                logger;
  std::unique_ptr<nvinfer1::IRuntime>      runtime;
  std::unique_ptr<nvinfer1::ICudaEngine>   engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::string                              input_name;
  std::string                              output_name;
  int                                      output_channels = 0;
  int                                      output_anchors  = 0;
  cudaStream_t                             stream = nullptr;
  void*                                    d_in   = nullptr;
  void*                                    d_out  = nullptr;
  size_t                                   d_in_size  = 0;
  size_t                                   d_out_size = 0;
};

TrtPredictor::TrtPredictor(const std::string& engine_path, int imgsz,
                           int max_batch,
                           std::string input_name, std::string output_name)
    : impl_(std::make_unique<Impl>()), imgsz_(imgsz), max_batch_(max_batch) {
  initLibNvInferPlugins(&impl_->logger, "");
  impl_->input_name  = std::move(input_name);
  impl_->output_name = std::move(output_name);

  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");
  impl_->engine.reset(
      impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine)
    throw std::runtime_error("deserializeCudaEngine failed for " + engine_path);
  impl_->ctx.reset(impl_->engine->createExecutionContext());
  if (!impl_->ctx) throw std::runtime_error("createExecutionContext failed");

  // Self-configure imgsz from the engine's static spatial dims when the engine
  // declares them (e.g. v4 builds at 608, v6-P6 at 1280, obb at 1024). Callers
  // pass a default (640) that won't match those engines; using the engine's
  // own H/W keeps letterbox + setInputShape consistent with how it was built.
  // Only override when the spatial dims are static (>0); a dynamic spatial dim
  // (-1) keeps the requested imgsz.
  {
    auto in_dims = impl_->engine->getTensorShape(impl_->input_name.c_str());
    if (in_dims.nbDims == 4 && in_dims.d[2] > 0 && in_dims.d[3] > 0 &&
        in_dims.d[2] == in_dims.d[3]) {
      imgsz_ = (int)in_dims.d[2];
    }
  }

  // Set runtime input shape to the requested max batch at the engine's imgsz.
  // Engine's profile kMAX must cover this batch; if it doesn't, setInputShape
  // returns false and we surface that to the caller.
  if (!impl_->ctx->setInputShape(
          impl_->input_name.c_str(),
          nvinfer1::Dims4(max_batch_, 3, imgsz_, imgsz_))) {
    throw std::runtime_error(
        "setInputShape rejected batch=" + std::to_string(max_batch_) +
        " at imgsz=" + std::to_string(imgsz_) +
        " — engine profile does not cover this shape");
  }

  CUDA_OK(cudaStreamCreate(&impl_->stream));
  impl_->d_in_size = (size_t)max_batch_ * 3 * imgsz_ * imgsz_ * sizeof(float);
  CUDA_OK(cudaMalloc(&impl_->d_in,  impl_->d_in_size));

  // Probe output shape (now resolved after setInputShape).
  auto out_dims = impl_->ctx->getTensorShape(impl_->output_name.c_str());
  if (out_dims.nbDims != 3 || out_dims.d[0] != max_batch_)
    throw std::runtime_error("unexpected output rank/shape from engine");
  impl_->output_channels = (int)out_dims.d[1];
  impl_->output_anchors  = (int)out_dims.d[2];
  impl_->d_out_size = (size_t)max_batch_ *
                      (size_t)impl_->output_channels *
                      (size_t)impl_->output_anchors * sizeof(float);
  CUDA_OK(cudaMalloc(&impl_->d_out, impl_->d_out_size));

  impl_->ctx->setTensorAddress(impl_->input_name.c_str(),  impl_->d_in);
  impl_->ctx->setTensorAddress(impl_->output_name.c_str(), impl_->d_out);

  std::cout << "[trt-pred] engine ready: in=" << impl_->input_name
            << "[" << max_batch_ << ",3," << imgsz_ << "," << imgsz_
            << "] → out=" << impl_->output_name
            << "[" << max_batch_ << "," << impl_->output_channels << ","
            << impl_->output_anchors << "]\n";
}

TrtPredictor::~TrtPredictor() {
  if (!impl_) return;
  if (impl_->stream) cudaStreamDestroy(impl_->stream);
  if (impl_->d_in)   cudaFree(impl_->d_in);
  if (impl_->d_out)  cudaFree(impl_->d_out);
}

std::vector<Detection> TrtPredictor::predict(const cv::Mat& bgr,
                                              NMSConfig nmscfg) const {
  auto batched = predict_batch({bgr}, nmscfg);
  return batched.empty() ? std::vector<Detection>{} : std::move(batched[0]);
}

std::vector<std::vector<Detection>>
TrtPredictor::predict_batch(const std::vector<cv::Mat>& bgrs,
                            NMSConfig nmscfg) const {
  const int N = (int)bgrs.size();
  if (N == 0) return {};
  if (N > max_batch_)
    throw std::runtime_error(
        "predict_batch: requested batch=" + std::to_string(N) +
        " exceeds engine max_batch=" + std::to_string(max_batch_));

  using ::yolocpp::core::ProfileScope;

  // GPU letterbox is on by default (#95C). A/B verified across
  // yolo8n / yolo11n / yolo11x: +28%, +18%, +9% respectively
  // — combined CPU letterbox + image_to_tensor + H2D (~0.46 ms)
  // collapses to a single ~0.25 ms gpu_letterbox pass. Set
  // YOLOCPP_GPU_LETTERBOX=0 to fall back to the CPU pipeline (kept
  // as the conservative path in case a future regression surfaces).
  static const bool gpu_lb =
      [] {
        const char* e = std::getenv("YOLOCPP_GPU_LETTERBOX");
        return !(e && e[0] == '0');                              // default ON
      }();

  std::vector<LetterboxResult> lbs;
  torch::Tensor x;
  if (gpu_lb) {
    GpuLetterboxBatch batch;
    {
      ProfileScope _s{"gpu_letterbox"};
      batch = gpu_letterbox_batch(bgrs, imgsz_,
                                  torch::Device(torch::kCUDA, 0));
      if (::yolocpp::core::Profile::instance().enabled())
        cudaStreamSynchronize(impl_->stream);
    }
    x   = std::move(batch.x);
    lbs = std::move(batch.lbs);
  } else {
    lbs.reserve(N);
    {
      ProfileScope _s{"letterbox"};
      for (const auto& bgr : bgrs) lbs.push_back(letterbox(bgr, imgsz_));
    }
    // Profiling (#95-followup) showed image_to_tensor (BGR→RGB +
    // uint8→float + /255 + HWC→CHW on CPU) was 51% of yolo11n's call
    // time. Match Ultralytics' approach (engine/predictor.py:163-176):
    // keep the CPU conversion as uint8 (~4× smaller H2D payload), then
    // cast to float + divide by 255 on GPU after the memcpy.
    std::vector<torch::Tensor> per_image;
    per_image.reserve(N);
    {
      ProfileScope _s{"image_to_tensor"};
      for (const auto& lb : lbs)
        per_image.push_back(image_to_tensor_u8_bgr_chw(lb.img));   // BGR CHW uint8
    }
    {
      ProfileScope _s{"stack+H2D"};
      auto x_u8 = torch::stack(per_image, /*dim=*/0).contiguous(); // [N, 3, H, W] BGR uint8 CPU
      x = x_u8.to(at::kCUDA, /*non_blocking=*/false)
              .flip(/*dim=*/1)
              .to(at::kFloat).div_(255.0f);
      if (::yolocpp::core::Profile::instance().enabled())
        cudaStreamSynchronize(impl_->stream);
    }
  }
  // We no longer need the explicit H2D below — the .to(kCUDA) above
  // already uploaded the bytes into a torch tensor. Hand TRT a pointer
  // to that tensor instead of memcpying through impl_->d_in.

  if (N != max_batch_) {
    if (!impl_->ctx->setInputShape(
            impl_->input_name.c_str(),
            nvinfer1::Dims4(N, 3, imgsz_, imgsz_))) {
      throw std::runtime_error(
          "setInputShape failed for batch=" + std::to_string(N));
    }
  }

  {
    ProfileScope _s{"enqueueV3"};
    // Zero-copy input: x is already a CUDA tensor (uint8→float on GPU
    // above). Point TRT's input binding straight at it instead of
    // memcpy'ing into impl_->d_in. Mirrors Ultralytics' approach
    // (nn/backends/tensorrt.py:141 — `int(im.data_ptr())`).
    impl_->ctx->setTensorAddress(impl_->input_name.c_str(), x.data_ptr());
    if (!impl_->ctx->enqueueV3(impl_->stream))
      throw std::runtime_error("enqueueV3 failed");
    CUDA_OK(cudaStreamSynchronize(impl_->stream));
    // Restore the persistent input binding so subsequent calls that
    // don't go through this fast path still find a valid pointer.
    impl_->ctx->setTensorAddress(impl_->input_name.c_str(), impl_->d_in);
  }

  // Alias the engine's CUDA output buffer as a torch tensor — no D2H
  // memcpy. The downstream `nms()` runs the [N, 4+nc, A] → conf filter
  // → top-k pipeline on GPU and only transfers the tiny survivor set
  // (typically K ≤ 100) to CPU for the AVX2 IoU loop. Ultralytics'
  // TRT backend does the same thing via `Binding.data` (also GPU
  // tensor aliasing the engine buffer). (Task #95 perf fix.)
  auto out_cuda = at::from_blob(
      impl_->d_out,
      {(int64_t)N, (int64_t)impl_->output_channels,
       (int64_t)impl_->output_anchors},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  std::vector<torch::Tensor> outs;
  {
    ProfileScope _s{"nms"};
    outs = nms(out_cuda, nmscfg);
  }
  ProfileScope _s_post{"postprocess"};
  std::vector<std::vector<Detection>> result(N);
  for (int i = 0; i < N && i < (int)outs.size(); ++i) {
    auto det = outs[i];
    if (det.size(0) == 0) continue;
    auto boxes = det.slice(1, 0, 4).contiguous();
    scale_boxes(boxes, lbs[i]);
    det.slice(1, 0, 4) = boxes;
    auto a = det.accessor<float, 2>();
    result[i].reserve(det.size(0));
    for (int j = 0; j < det.size(0); ++j) {
      Detection d;
      d.x1   = a[j][0]; d.y1 = a[j][1]; d.x2 = a[j][2]; d.y2 = a[j][3];
      d.conf = a[j][4]; d.cls = (int)a[j][5];
      result[i].push_back(d);
    }
  }
  return result;
}

Results TrtPredictor::predict_results(const cv::Mat& bgr,
                                      const std::vector<std::string>& names,
                                      NMSConfig conf) const {
  auto batched = predict_results_batch({bgr}, names, conf);
  return batched.empty() ? Results{} : std::move(batched[0]);
}

std::vector<Results>
TrtPredictor::predict_results_batch(const std::vector<cv::Mat>& bgrs,
                                    const std::vector<std::string>& names,
                                    NMSConfig conf) const {
  // Time the whole pipeline: preprocess wall + inference wall +
  // postprocess wall (best-effort — uses the same chrono breaks the
  // --profile mode shows but doesn't depend on it being enabled).
  using clk = std::chrono::steady_clock;
  using ms  = std::chrono::duration<double, std::milli>;
  auto t0 = clk::now();
  auto per_image_dets = predict_batch(bgrs, conf);
  auto t1 = clk::now();
  double total_ms = ms(t1 - t0).count();

  std::vector<Results> out;
  out.reserve(bgrs.size());
  for (std::size_t i = 0; i < bgrs.size(); ++i) {
    Results r;
    r.boxes    = (i < per_image_dets.size()) ? std::move(per_image_dets[i])
                                             : std::vector<Detection>{};
    r.orig_img = bgrs[i];
    r.orig_w   = bgrs[i].cols;
    r.orig_h   = bgrs[i].rows;
    r.names    = names;
    // Split the bulk timing across phases by share. Real per-phase
    // numbers come from --profile; this matches Ultralytics' speed
    // dict (a coarse triple, not exact).
    r.speed.inference_ms  = total_ms / (double)bgrs.size();
    r.speed.preprocess_ms = 0.0;
    r.speed.postprocess_ms = 0.0;
    out.push_back(std::move(r));
  }
  return out;
}

std::vector<Detection> TrtPredictor::predict_to_file(
    const std::string& in_path, const std::string& out_path, NMSConfig nmscfg,
    const std::vector<std::string>& names) const {
  cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
  if (img.empty())
    throw std::runtime_error("could not read image: " + in_path);
  auto dets = predict(img, nmscfg);
  const auto& nm = names.empty() ? coco_names() : names;
  for (const auto& d : dets) {
    cv::Scalar color{(double)((d.cls * 41) % 256),
                     (double)((d.cls * 73) % 256),
                     (double)((d.cls * 11) % 256)};
    cv::rectangle(img, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2},
                  color, 2);
    std::string label = (d.cls >= 0 && d.cls < (int)nm.size()
                             ? nm[d.cls] : std::to_string(d.cls));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
    cv::putText(img, buf, {(int)d.x1 + 2, (int)d.y1 + 14},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  if (!cv::imwrite(out_path, img))
    throw std::runtime_error("could not write image: " + out_path);
  return dets;
}

}  // namespace yolocpp::inference
