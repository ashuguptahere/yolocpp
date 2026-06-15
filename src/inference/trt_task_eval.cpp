#include "yolocpp/inference/trt_task_eval.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

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

struct OutBuf {
  std::string          name;
  std::vector<int64_t> dims;
  void*                buf  = nullptr;
  size_t               size = 0;
};

// Holds the engine + its bound output buffers. Kept alive by the returned
// closure's capture; the destructor frees the CUDA resources.
struct EngineState {
  TrtLogger                                    logger;
  std::unique_ptr<nvinfer1::IRuntime>          runtime;
  std::unique_ptr<nvinfer1::ICudaEngine>       engine;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::string                                  input_name;
  cudaStream_t                                 stream = nullptr;
  std::vector<OutBuf>                          outputs;
  ~EngineState() {
    if (stream) cudaStreamDestroy(stream);
    for (auto& o : outputs)
      if (o.buf) cudaFree(o.buf);
  }
};

}  // namespace

TrtMultiForward make_trt_multi_forward(const std::string& engine_path,
                                       int& imgsz) {
  auto st = std::make_shared<EngineState>();
  initLibNvInferPlugins(&st->logger, "");

  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  st->runtime.reset(nvinfer1::createInferRuntime(st->logger));
  if (!st->runtime) throw std::runtime_error("createInferRuntime failed");
  st->engine.reset(
      st->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!st->engine)
    throw std::runtime_error("deserializeCudaEngine failed for " + engine_path);
  st->ctx.reset(st->engine->createExecutionContext());
  if (!st->ctx) throw std::runtime_error("createExecutionContext failed");

  // Identify the single input tensor; collect output tensor names.
  std::vector<std::string> out_names;
  for (int i = 0; i < st->engine->getNbIOTensors(); ++i) {
    const char* nm = st->engine->getIOTensorName(i);
    if (st->engine->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT)
      st->input_name = nm;
    else
      out_names.emplace_back(nm);
  }
  if (st->input_name.empty()) throw std::runtime_error("engine has no input");

  // Prefer the engine's static spatial dims (obb=1024, v6-P6=1280, …) over the
  // caller's default; keep the requested imgsz when the dim is dynamic (-1).
  {
    auto in_dims = st->engine->getTensorShape(st->input_name.c_str());
    if (in_dims.nbDims == 4 && in_dims.d[2] > 0 && in_dims.d[3] > 0 &&
        in_dims.d[2] == in_dims.d[3])
      imgsz = (int)in_dims.d[2];
  }
  if (!st->ctx->setInputShape(st->input_name.c_str(),
                              nvinfer1::Dims4(1, 3, imgsz, imgsz)))
    throw std::runtime_error(
        "setInputShape rejected 1×3×" + std::to_string(imgsz) + "² — engine "
        "profile does not cover this shape");

  CUDA_OK(cudaStreamCreate(&st->stream));

  // Allocate + bind every output buffer at the now-resolved shapes.
  for (const auto& nm : out_names) {
    auto d = st->ctx->getTensorShape(nm.c_str());
    OutBuf ob;
    ob.name = nm;
    ob.dims.assign(d.d, d.d + d.nbDims);
    int64_t numel =
        std::accumulate(ob.dims.begin(), ob.dims.end(), (int64_t)1,
                        std::multiplies<int64_t>());
    ob.size = (size_t)numel * sizeof(float);
    CUDA_OK(cudaMalloc(&ob.buf, ob.size));
    st->ctx->setTensorAddress(nm.c_str(), ob.buf);
    st->outputs.push_back(std::move(ob));
  }

  return [st](torch::Tensor x) -> std::map<std::string, torch::Tensor> {
    auto xc = x.to(at::kCUDA).to(at::kFloat).contiguous();
    st->ctx->setTensorAddress(st->input_name.c_str(), xc.data_ptr());
    if (!st->ctx->enqueueV3(st->stream))
      throw std::runtime_error("enqueueV3 failed");
    CUDA_OK(cudaStreamSynchronize(st->stream));
    std::map<std::string, torch::Tensor> out;
    for (const auto& o : st->outputs) {
      // Clone out of the persistent engine buffer (overwritten next call).
      auto t = at::from_blob(
                   o.buf, o.dims,
                   at::TensorOptions().dtype(at::kFloat).device(at::kCUDA))
                   .clone();
      out.emplace(o.name, std::move(t));
    }
    return out;
  };
}

}  // namespace yolocpp::inference
