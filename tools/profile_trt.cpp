// One-off profiler: time each phase of the TRT inference pipeline so
// we know where the cycles actually go. Run with:
//
//   profile_trt <engine.trt> [imgsz=640] [iters=30]
//
// Splits per-call work into:
//   t_letterbox  — CPU letterbox (resize + pad)
//   t_to_tensor  — image_to_tensor (BGR→RGB float CHW)
//   t_h2d        — cudaMemcpyAsync H2D + sync
//   t_enqueue    — enqueueV3 + sync (pure GPU compute)
//   t_nms        — nms() (conf filter, top-k, IoU loop)
// Total = sum + small overhead. Reports median across `iters`.

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/nms.hpp"

using clk = std::chrono::steady_clock;
using ms  = std::chrono::duration<double, std::milli>;

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kERROR) std::cerr << "[trt] " << msg << "\n";
  }
};

#define CUDA_OK(call) do { auto _e=(call); if(_e!=cudaSuccess) { \
  std::cerr<<"cuda: "<<cudaGetErrorString(_e)<<"\n"; std::exit(1); }} while(0)

double med(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return xs.empty() ? 0.0 : xs[xs.size()/2];
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: profile_trt <engine.trt> [imgsz=640] [iters=30]\n";
    return 1;
  }
  std::string engine_path = argv[1];
  int imgsz = (argc > 2) ? std::atoi(argv[2]) : 640;
  int iters = (argc > 3) ? std::atoi(argv[3]) : 30;
  int warmup = 5;

  TrtLogger logger;
  initLibNvInferPlugins(&logger, "");

  // Load engine
  std::ifstream f(engine_path, std::ios::binary);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(blob.data(), blob.size()));
  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());

  ctx->setInputShape("images", nvinfer1::Dims4(1, 3, imgsz, imgsz));
  cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));

  size_t in_bytes = 1ull * 3 * imgsz * imgsz * sizeof(float);
  auto out_dims = ctx->getTensorShape("output");
  size_t out_elems = 1;
  for (int i = 0; i < out_dims.nbDims; ++i) out_elems *= out_dims.d[i];
  size_t out_bytes = out_elems * sizeof(float);
  void *d_in, *d_out;
  CUDA_OK(cudaMalloc(&d_in, in_bytes));
  CUDA_OK(cudaMalloc(&d_out, out_bytes));
  ctx->setTensorAddress("images", d_in);
  ctx->setTensorAddress("output", d_out);

  cv::Mat img = cv::imread("/tmp/bus.jpg");
  if (img.empty()) { std::cerr << "cannot read /tmp/bus.jpg\n"; return 1; }

  std::vector<double> t_lb, t_tens, t_h2d, t_enq, t_nms;

  for (int i = 0; i < warmup + iters; ++i) {
    auto t0 = clk::now();
    auto lb = yolocpp::inference::letterbox(img, imgsz);
    auto t1 = clk::now();
    // New path: uint8 CHW (no float convert on CPU).
    auto x_u8 = yolocpp::inference::image_to_tensor_u8(lb.img).unsqueeze(0).contiguous();
    auto t2 = clk::now();
    // H2D of uint8 (4× smaller) + GPU cast to float + /255.
    auto x = x_u8.to(at::kCUDA).to(at::kFloat).div_(255.0f);
    cudaStreamSynchronize(stream);
    auto t3 = clk::now();
    ctx->setTensorAddress("images", x.data_ptr());
    ctx->enqueueV3(stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    auto t4 = clk::now();
    auto out_cuda = at::from_blob(
        d_out, {(int64_t)out_dims.d[0], (int64_t)out_dims.d[1], (int64_t)out_dims.d[2]},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    auto dets = yolocpp::inference::nms(out_cuda, {});
    auto t5 = clk::now();

    if (i >= warmup) {
      t_lb.push_back(ms(t1-t0).count());
      t_tens.push_back(ms(t2-t1).count());
      t_h2d.push_back(ms(t3-t2).count());
      t_enq.push_back(ms(t4-t3).count());
      t_nms.push_back(ms(t5-t4).count());
    }
  }

  double m_lb = med(t_lb), m_tens = med(t_tens), m_h2d = med(t_h2d);
  double m_enq = med(t_enq), m_nms = med(t_nms);
  double total = m_lb + m_tens + m_h2d + m_enq + m_nms;
  auto pct = [&](double t) { return 100.0 * t / total; };

  printf("\n=== %s  iters=%d ===\n", engine_path.c_str(), iters);
  printf("  phase            median(ms)   %%\n");
  printf("  ──────────────  ───────────  ─────\n");
  printf("  letterbox       %8.3f      %5.1f\n", m_lb,   pct(m_lb));
  printf("  image_to_tensor %8.3f      %5.1f\n", m_tens, pct(m_tens));
  printf("  H2D + sync      %8.3f      %5.1f\n", m_h2d,  pct(m_h2d));
  printf("  enqueueV3+sync  %8.3f      %5.1f\n", m_enq,  pct(m_enq));
  printf("  nms             %8.3f      %5.1f\n", m_nms,  pct(m_nms));
  printf("  ──────────────  ───────────  ─────\n");
  printf("  TOTAL           %8.3f      100.0\n", total);
  printf("  throughput:     %8.1f img/s\n", 1000.0 / total);

  CUDA_OK(cudaFree(d_in)); CUDA_OK(cudaFree(d_out));
  cudaStreamDestroy(stream);
  return 0;
}
