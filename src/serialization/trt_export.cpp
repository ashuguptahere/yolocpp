#include "yolocpp/serialization/trt_export.hpp"
#include "yolocpp/inference/letterbox.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace yolocpp::serialization {

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
};

// INT8 entropy calibrator. Reads JPG/PNG images from `dir`, letterboxes
// each one to imgsz×imgsz BGR→RGB float [0,1], uploads `batch_size`
// images at a time to a single GPU buffer. `cache_path` (if non-empty)
// stores/loads the calibration histogram so repeated builds skip
// re-sampling — non-trivial speedup since calibration walks the full
// dataset through the engine in INT8-search mode.
class ImgDirCalibrator final : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  ImgDirCalibrator(const std::string& dir, int imgsz, int batch_size,
                   std::string cache_path)
      : imgsz_(imgsz), batch_size_(batch_size),
        cache_path_(std::move(cache_path)) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
      throw std::runtime_error(
          "INT8 calibration directory does not exist: " + dir);
    }
    for (const auto& e : fs::directory_iterator(dir)) {
      const auto ext = e.path().extension().string();
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
        files_.push_back(e.path().string());
      }
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) {
      throw std::runtime_error("INT8 calibration: no images in " + dir);
    }
    const std::size_t per_image = (std::size_t)3 * imgsz_ * imgsz_ * sizeof(float);
    cudaError_t err = cudaMalloc(&d_input_, (std::size_t)batch_size_ * per_image);
    if (err != cudaSuccess) {
      throw std::runtime_error("INT8 cudaMalloc failed");
    }
    std::cerr << "[trt-int8] calibrating on " << files_.size()
              << " images (batch=" << batch_size_ << ") from " << dir << "\n";
  }

  ~ImgDirCalibrator() override {
    if (d_input_) cudaFree(d_input_);
  }

  int getBatchSize() const noexcept override { return batch_size_; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override {
    if (idx_ + (std::size_t)batch_size_ > files_.size()) return false;

    const std::size_t per = (std::size_t)3 * imgsz_ * imgsz_;
    std::vector<float> h_buf((std::size_t)batch_size_ * per);
    for (int b = 0; b < batch_size_; ++b) {
      cv::Mat im = cv::imread(files_[idx_ + (std::size_t)b], cv::IMREAD_COLOR);
      if (im.empty()) {
        std::cerr << "[trt-int8] WARN: failed to read "
                  << files_[idx_ + (std::size_t)b] << ", skipping batch\n";
        return false;
      }
      auto lb = inference::letterbox(im, imgsz_,
                                      /*pad_color=*/cv::Scalar(114, 114, 114),
                                      /*scale_up=*/false,
                                      /*auto_minrec=*/false);
      cv::Mat rgb;
      cv::cvtColor(lb.img, rgb, cv::COLOR_BGR2RGB);
      // HWC uint8 → CHW float [0,1].
      const int H = rgb.rows, W = rgb.cols;
      const uchar* p = rgb.data;
      float* dst = h_buf.data() + (std::size_t)b * per;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const uchar* px = p + ((std::size_t)y * W + x) * 3;
          dst[0 * H * W + y * W + x] = px[0] / 255.0f;
          dst[1 * H * W + y * W + x] = px[1] / 255.0f;
          dst[2 * H * W + y * W + x] = px[2] / 255.0f;
        }
      }
    }
    cudaError_t err = cudaMemcpy(d_input_, h_buf.data(),
                                  h_buf.size() * sizeof(float),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "[trt-int8] cudaMemcpy failed\n";
      return false;
    }
    bindings[0] = d_input_;
    idx_ += (std::size_t)batch_size_;
    if ((idx_ / (std::size_t)batch_size_) % 10 == 0) {
      std::cerr << "[trt-int8] calibrated " << idx_ << "/" << files_.size()
                << "\n";
    }
    return true;
  }

  const void* readCalibrationCache(std::size_t& length) noexcept override {
    if (cache_path_.empty()) { length = 0; return nullptr; }
    std::ifstream f(cache_path_, std::ios::binary);
    if (!f) { length = 0; return nullptr; }
    cache_buf_.assign(std::istreambuf_iterator<char>(f), {});
    length = cache_buf_.size();
    return cache_buf_.data();
  }

  void writeCalibrationCache(const void* ptr,
                             std::size_t length) noexcept override {
    if (cache_path_.empty()) return;
    std::ofstream f(cache_path_, std::ios::binary);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(ptr), (std::streamsize)length);
  }

 private:
  int imgsz_;
  int batch_size_;
  std::vector<std::string> files_;
  std::size_t idx_ = 0;
  void* d_input_ = nullptr;
  std::string cache_path_;
  std::vector<char> cache_buf_;
};

}  // anonymous namespace

void build_trt_engine(const std::string& onnx_path,
                      const std::string& engine_path,
                      const TrtBuildConfig& cfg) {
  TrtLogger logger;
  initLibNvInferPlugins(&logger, "");

  std::unique_ptr<nvinfer1::IBuilder> builder{
      nvinfer1::createInferBuilder(logger)};
  if (!builder) throw std::runtime_error("createInferBuilder failed");

  std::unique_ptr<nvinfer1::INetworkDefinition> network{
      builder->createNetworkV2(0U)};  // explicit batch (TRT 10 default)
  if (!network) throw std::runtime_error("createNetworkV2 failed");

  std::unique_ptr<nvonnxparser::IParser> parser{
      nvonnxparser::createParser(*network, logger)};
  if (!parser) throw std::runtime_error("createParser failed");

  if (!parser->parseFromFile(
          onnx_path.c_str(),
          (int)nvinfer1::ILogger::Severity::kWARNING)) {
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      std::cerr << "[trt-parse] " << parser->getError(i)->desc() << "\n";
    }
    throw std::runtime_error("nvonnxparser failed to parse " + onnx_path);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> bcfg{
      builder->createBuilderConfig()};
  bcfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                           cfg.workspace_bytes);
  if (cfg.fp16 && builder->platformHasFastFp16()) {
    bcfg->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  std::unique_ptr<ImgDirCalibrator> calibrator;
  if (cfg.int8 && builder->platformHasFastInt8()) {
    if (cfg.calib_image_dir.empty()) {
      throw std::runtime_error(
          "INT8 build requested but calib_image_dir is empty");
    }
    bcfg->setFlag(nvinfer1::BuilderFlag::kINT8);
    // INT8 calibration uses the OPT batch from the profile below; cap
    // at 8 to keep memory sane on small workspaces.
    const int cal_batch = std::min(cfg.batch_opt, 8);
    calibrator = std::make_unique<ImgDirCalibrator>(
        cfg.calib_image_dir, cfg.imgsz, cal_batch, cfg.calib_cache);
    bcfg->setInt8Calibrator(calibrator.get());
  }
  if (!cfg.tf32) {
    // TRT enables TF32 by default in FP32 builds; clearing it forces true
    // FP32 math through the tensor cores. Required for v10 s/m/b/l/x.
    bcfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
  }
  bcfg->setBuilderOptimizationLevel(cfg.builder_opt_level);

  // Optimization profile for our dynamic batch dimension.
  auto* profile = builder->createOptimizationProfile();
  profile->setDimensions(cfg.input_name.c_str(),
                         nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims4(cfg.batch_min, 3, cfg.imgsz, cfg.imgsz));
  profile->setDimensions(cfg.input_name.c_str(),
                         nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims4(cfg.batch_opt, 3, cfg.imgsz, cfg.imgsz));
  profile->setDimensions(cfg.input_name.c_str(),
                         nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims4(cfg.batch_max, 3, cfg.imgsz, cfg.imgsz));
  bcfg->addOptimizationProfile(profile);

  std::cerr << "[trt] building engine (this may take a couple minutes)...\n";
  std::unique_ptr<nvinfer1::IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *bcfg)};
  if (!plan) throw std::runtime_error("buildSerializedNetwork returned null");

  std::ofstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + engine_path);
  f.write(reinterpret_cast<const char*>(plan->data()),
          (std::streamsize)plan->size());
  std::cerr << "[trt] wrote " << engine_path << " (" << plan->size() << " bytes)\n";
}

}  // namespace yolocpp::serialization
