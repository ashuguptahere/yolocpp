#include "yolocpp/inference/trt_predictor.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "yolocpp/inference/letterbox.hpp"

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

  // Set runtime input shape to the requested max batch. Engine's profile
  // kMAX must cover this; if it doesn't, setInputShape returns false and
  // we surface that to the caller.
  if (!impl_->ctx->setInputShape(
          impl_->input_name.c_str(),
          nvinfer1::Dims4(max_batch_, 3, imgsz, imgsz))) {
    throw std::runtime_error(
        "setInputShape rejected batch=" + std::to_string(max_batch_) +
        " — engine profile kMAX is smaller than requested batch");
  }

  CUDA_OK(cudaStreamCreate(&impl_->stream));
  impl_->d_in_size = (size_t)max_batch_ * 3 * imgsz * imgsz * sizeof(float);
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
            << "[" << max_batch_ << ",3," << imgsz << "," << imgsz
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

  std::vector<LetterboxResult> lbs;
  lbs.reserve(N);
  for (const auto& bgr : bgrs) lbs.push_back(letterbox(bgr, imgsz_));

  std::vector<torch::Tensor> per_image;
  per_image.reserve(N);
  for (const auto& lb : lbs) per_image.push_back(image_to_tensor(lb.img));
  auto x = torch::stack(per_image, /*dim=*/0).contiguous();   // [N, 3, H, W]

  if (N != max_batch_) {
    if (!impl_->ctx->setInputShape(
            impl_->input_name.c_str(),
            nvinfer1::Dims4(N, 3, imgsz_, imgsz_))) {
      throw std::runtime_error(
          "setInputShape failed for batch=" + std::to_string(N));
    }
  }

  const size_t in_bytes  = (size_t)N * 3 * imgsz_ * imgsz_ * sizeof(float);

  CUDA_OK(cudaMemcpyAsync(impl_->d_in, x.data_ptr(), in_bytes,
                          cudaMemcpyHostToDevice, impl_->stream));
  if (!impl_->ctx->enqueueV3(impl_->stream))
    throw std::runtime_error("enqueueV3 failed");
  CUDA_OK(cudaStreamSynchronize(impl_->stream));

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

  auto outs = nms(out_cuda, nmscfg);
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
