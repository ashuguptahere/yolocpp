#include "yolocpp/serialization/trt_export.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace yolocpp::serialization {

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
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
