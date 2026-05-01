#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/inference/letterbox.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <fstream>

using namespace yolocpp::inference;

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: dump_trt engine.trt image.jpg [imgsz]\n"; return 1; }
  std::string engine_path = argv[1];
  std::string image_path  = argv[2];
  int imgsz = (argc > 3) ? std::stoi(argv[3]) : 640;

  class L : public nvinfer1::ILogger {
   public: void log(Severity s, const char* msg) noexcept override {
     if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
   }
  } logger;
  initLibNvInferPlugins(&logger, "");

  std::ifstream f(engine_path, std::ios::binary);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  std::unique_ptr<nvinfer1::IRuntime> rt{nvinfer1::createInferRuntime(logger)};
  std::unique_ptr<nvinfer1::ICudaEngine> eng{rt->deserializeCudaEngine(blob.data(), blob.size())};
  std::unique_ptr<nvinfer1::IExecutionContext> ctx{eng->createExecutionContext()};

  ctx->setInputShape("images", nvinfer1::Dims4(1, 3, imgsz, imgsz));
  auto out_dims = ctx->getTensorShape("output");
  int C = out_dims.d[1], A = out_dims.d[2];
  std::cerr << "out shape [1," << C << "," << A << "]\n";

  cv::Mat img = cv::imread(image_path);
  auto lb = letterbox(img, imgsz);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).contiguous();

  void *d_in, *d_out;
  cudaMalloc(&d_in, 1*3*imgsz*imgsz*sizeof(float));
  cudaMalloc(&d_out, C*A*sizeof(float));
  cudaMemcpy(d_in, x.data_ptr(), 1*3*imgsz*imgsz*sizeof(float), cudaMemcpyHostToDevice);
  cudaStream_t s; cudaStreamCreate(&s);
  ctx->setTensorAddress("images", d_in);
  ctx->setTensorAddress("output", d_out);
  ctx->enqueueV3(s);

  auto out_cpu = torch::empty({1, C, A}, torch::kFloat32);
  cudaMemcpyAsync(out_cpu.data_ptr(), d_out, C*A*sizeof(float), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  auto pred = out_cpu[0]; // [C, A]
  auto cls = pred.slice(0, 4, C); // [nc, A]
  auto conf = std::get<0>(cls.max(0)); // [A]
  auto sorted = std::get<0>(conf.sort(0, true));
  std::cerr << "top10 conf:";
  for (int i = 0; i < 10 && i < (int)sorted.size(0); ++i) std::cerr << " " << sorted[i].item<float>();
  std::cerr << "\n";
  auto box = pred.slice(0, 0, 4); // [4, A]
  auto box0 = box.slice(1, 0, 5).t().contiguous();  // [5, 4]
  std::cerr << "first 5 boxes:\n";
  auto a = box0.accessor<float, 2>();
  for (int i = 0; i < 5; ++i) std::cerr << "  [" << a[i][0] << "," << a[i][1] << "," << a[i][2] << "," << a[i][3] << "]\n";

  cudaFree(d_in); cudaFree(d_out); cudaStreamDestroy(s);
  return 0;
}
