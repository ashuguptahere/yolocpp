// End-to-end smoke test: every dependency is exercised, GPU is required.
//
//   - LibTorch     : create a tensor on CUDA, run a matmul, copy to host
//   - Custom CUDA  : launch add_kernel from smoke_kernel.cu
//   - OpenCV       : encode/decode an in-memory JPEG round-trip
//   - TensorRT     : build a tiny Identity network natively (no ONNX),
//                    serialize + deserialize an engine, run one inference,
//                    create an ONNX parser instance to verify nvonnxparser
//                    is linked and loadable.
//
// Exit code 0 means everything works.

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolocpp/core/device.hpp"
#include "yolocpp/core/version.hpp"

namespace yolocpp_smoke {
cudaError_t launch_add(const float* a, const float* b, float* c, int n);
}

#define EXPECT(cond, msg)                                                 \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "[FAIL] " << msg << "\n";                              \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
};

bool test_libtorch_cuda() {
  std::cout << "[libtorch] cuda::is_available = "
            << (torch::cuda::is_available() ? "yes" : "no") << "\n";
  if (!torch::cuda::is_available()) return false;
  auto a = torch::randn({256, 256}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  auto b = torch::randn({256, 256}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  auto c = a.matmul(b).to(torch::kCPU);
  std::cout << "[libtorch] matmul ok, c[0,0]=" << c.index({0, 0}).item<float>() << "\n";
  return true;
}

bool test_custom_cuda_kernel() {
  const int n = 1 << 16;
  std::vector<float> ha(n, 1.f), hb(n, 2.f), hc(n, 0.f);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (cudaMalloc(&da, n * sizeof(float)) != cudaSuccess) return false;
  if (cudaMalloc(&db, n * sizeof(float)) != cudaSuccess) return false;
  if (cudaMalloc(&dc, n * sizeof(float)) != cudaSuccess) return false;
  cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaError_t e = yolocpp_smoke::launch_add(da, db, dc, n);
  if (e != cudaSuccess) { std::cerr << "kernel launch: " << cudaGetErrorString(e) << "\n"; return false; }
  cudaDeviceSynchronize();
  cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dc);
  for (int i = 0; i < n; ++i)
    if (hc[i] != 3.f) { std::cerr << "kernel result wrong at " << i << ": " << hc[i] << "\n"; return false; }
  std::cout << "[cuda] custom add kernel ok (" << n << " elements)\n";
  return true;
}

bool test_opencv_jpeg_roundtrip() {
  cv::Mat src(64, 64, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::circle(src, {32, 32}, 16, cv::Scalar(255, 255, 255), -1);
  std::vector<uchar> buf;
  if (!cv::imencode(".jpg", src, buf, {cv::IMWRITE_JPEG_QUALITY, 90})) return false;
  cv::Mat dst = cv::imdecode(buf, cv::IMREAD_COLOR);
  if (dst.empty() || dst.size() != src.size() || dst.type() != src.type()) return false;
  std::cout << "[opencv] jpeg encode/decode ok (" << buf.size() << " bytes, "
            << dst.cols << "x" << dst.rows << ")\n";
  return true;
}

// Build a TRT network natively: input -> Identity -> output.
bool test_tensorrt_native_roundtrip() {
  TrtLogger logger;

  // initLibNvInferPlugins forces a symbol from libnvinfer_plugin.so to load,
  // so this also validates that linkage.
  if (!initLibNvInferPlugins(&logger, "")) {
    std::cerr << "[trt] initLibNvInferPlugins returned false\n";
    return false;
  }

  std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
  if (!builder) return false;
  // TRT 10 dropped implicit-batch mode; createNetworkV2(0) gets explicit batch.
  std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
  if (!network) return false;

  auto* in = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, 4});
  if (!in) return false;
  auto* idl = network->addIdentity(*in);
  if (!idl) return false;
  idl->getOutput(0)->setName("y");
  network->markOutput(*idl->getOutput(0));

  std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
  if (!config) return false;
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ull << 28);

  std::unique_ptr<nvinfer1::IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) return false;

  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
  if (!runtime) return false;
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(plan->data(), plan->size())};
  if (!engine) return false;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx{engine->createExecutionContext()};
  if (!ctx) return false;

  if (engine->getNbIOTensors() < 2) return false;
  const char* in_name  = engine->getIOTensorName(0);
  const char* out_name = engine->getIOTensorName(1);

  std::vector<float> h_in{1.f, 2.f, 3.f, 4.f}, h_out(4, 0.f);
  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in,  h_in.size()  * sizeof(float));
  cudaMalloc(&d_out, h_out.size() * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);

  ctx->setInputShape(in_name, nvinfer1::Dims2{1, 4});
  ctx->setTensorAddress(in_name,  d_in);
  ctx->setTensorAddress(out_name, d_out);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  bool ok = ctx->enqueueV3(stream);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamDestroy(stream);
  cudaFree(d_in); cudaFree(d_out);
  if (!ok) return false;
  for (size_t i = 0; i < h_in.size(); ++i)
    if (h_out[i] != h_in[i]) {
      std::cerr << "[trt] output mismatch at " << i << "\n";
      return false;
    }
  std::cout << "[tensorrt] native build+infer ok\n";
  return true;
}

// Force-link nvonnxparser by constructing an empty parser instance.
bool test_tensorrt_onnx_parser_loadable() {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
  if (!builder) return false;
  // TRT 10 dropped implicit-batch mode; createNetworkV2(0) gets explicit batch.
  std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
  if (!network) return false;
  std::unique_ptr<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, logger)};
  if (!parser) {
    std::cerr << "[trt] createParser returned null\n";
    return false;
  }
  std::cout << "[tensorrt] nvonnxparser linked and instantiated ok\n";
  return true;
}

}  // namespace

int main() {
  const auto info = yolocpp::build_info();
  std::cout << "=== yolocpp smoke test ===\n";
  std::cout << "yolocpp     " << info.yolocpp_version       << "\n";
  std::cout << "libtorch    " << info.libtorch_version      << "\n";
  std::cout << "cuda_tk     " << info.cuda_toolkit_version  << "\n";
  std::cout << "cuda_rt     " << info.cuda_runtime_version  << "\n";
  std::cout << "tensorrt    " << info.tensorrt_version      << "\n";
  std::cout << "opencv      " << info.opencv_version        << "\n";

  for (const auto& d : yolocpp::list_cuda_devices())
    std::cout << "device[" << d.index << "] " << d.name
              << " sm_" << d.compute_capability_major << d.compute_capability_minor
              << " " << (d.total_memory_bytes >> 20) << " MiB, "
              << d.multi_processor_count << " SMs\n";

  EXPECT(test_libtorch_cuda(),                  "libtorch CUDA");
  EXPECT(test_custom_cuda_kernel(),             "custom CUDA kernel");
  EXPECT(test_opencv_jpeg_roundtrip(),          "OpenCV JPEG round-trip");
  EXPECT(test_tensorrt_native_roundtrip(),      "TensorRT native build+infer");
  EXPECT(test_tensorrt_onnx_parser_loadable(),  "TensorRT ONNX parser link");

  std::cout << "=== ALL PASS ===\n";
  return 0;
}
