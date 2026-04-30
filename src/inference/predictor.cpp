#include "yolocpp/inference/predictor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>

#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace yolocpp::inference {

const std::vector<std::string>& coco_names() {
  static const std::vector<std::string> n = {
      "person","bicycle","car","motorcycle","airplane","bus","train","truck",
      "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
      "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
      "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
      "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
      "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
      "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
      "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
      "potted plant","bed","dining table","toilet","tv","laptop","mouse",
      "remote","keyboard","cell phone","microwave","oven","toaster","sink",
      "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
      "toothbrush",
  };
  return n;
}

static torch::Device resolve_device(std::string s) {
  if (s.empty()) s = torch::cuda::is_available() ? "cuda" : "cpu";
  if (s == "cuda" || s.rfind("cuda:", 0) == 0)
    return torch::Device(torch::kCUDA, s == "cuda" ? 0 : std::stoi(s.substr(5)));
  return torch::Device(torch::kCPU);
}

Predictor::Predictor(const std::string& weights, int imgsz, std::string device,
                     int nc, models::Yolo8Scale scale)
    : model_(scale, nc),
      device_(resolve_device(std::move(device))),
      imgsz_(imgsz) {
  // Try Ultralytics .pt format first; fall back to torch::save archives
  // (which is what our trainer emits via `torch::save(ema_, ...)`).
  try {
    auto sd = serialization::load_state_dict(weights);
    int copied = model_->load_from_state_dict(sd.entries);
    std::cout << "[predictor] loaded " << copied
              << " tensors from " << weights << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[predictor] Ultralytics-format load failed (" << e.what()
              << "); trying torch::save format...\n";
    torch::load(model_, weights);
    std::cout << "[predictor] loaded torch::save archive: " << weights << "\n";
  }

  model_->to(device_);
  model_->eval();
}

std::vector<Detection> Predictor::predict(const cv::Mat& bgr,
                                          NMSConfig conf) const {
  auto lb = letterbox(bgr, imgsz_);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(device_);

  torch::Tensor pred;
  {
    torch::NoGradGuard ng;
    pred = const_cast<models::Yolo8Detect&>(model_)->forward_eval(x);
  }
  auto outs = nms(pred, conf);
  TORCH_CHECK(outs.size() == 1, "expected single image batch");
  auto& det = outs[0];   // [k, 6] xyxy + conf + cls (in letterbox coords)

  // Move to CPU and unscale.
  det = det.to(torch::kCPU);
  if (det.size(0) == 0) return {};

  auto boxes = det.slice(1, 0, 4).contiguous();
  scale_boxes(boxes, lb);
  det.slice(1, 0, 4) = boxes;

  std::vector<Detection> result;
  result.reserve(det.size(0));
  auto a = det.accessor<float, 2>();
  for (int i = 0; i < det.size(0); ++i) {
    Detection d;
    d.x1   = a[i][0]; d.y1 = a[i][1]; d.x2 = a[i][2]; d.y2 = a[i][3];
    d.conf = a[i][4];
    d.cls  = (int)a[i][5];
    result.push_back(d);
  }
  return result;
}

std::vector<Detection> Predictor::predict_to_file(
    const std::string& in_path, const std::string& out_path, NMSConfig conf,
    const std::vector<std::string>& names) const {
  cv::Mat img = cv::imread(in_path, cv::IMREAD_COLOR);
  if (img.empty())
    throw std::runtime_error("could not read image: " + in_path);

  auto dets = predict(img, conf);
  const auto& nm = names.empty() ? coco_names() : names;

  for (const auto& d : dets) {
    cv::Scalar color{
        (double)((d.cls * 41) % 256),
        (double)((d.cls * 73) % 256),
        (double)((d.cls * 11) % 256),
    };
    cv::rectangle(img, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2},
                  color, 2);
    std::string label = (d.cls >= 0 && d.cls < (int)nm.size() ? nm[d.cls]
                                                              : std::to_string(d.cls));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
    int baseline = 0;
    auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int yt = std::max((int)d.y1, sz.height + 4);
    cv::rectangle(img, {(int)d.x1, yt - sz.height - 4},
                  {(int)d.x1 + sz.width + 4, yt}, color, cv::FILLED);
    cv::putText(img, buf, {(int)d.x1 + 2, yt - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  if (!cv::imwrite(out_path, img))
    throw std::runtime_error("could not write image: " + out_path);
  return dets;
}

std::vector<Detection> predict_v5_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz, const std::string& device,
    int nc, models::Yolo8Scale scale, NMSConfig nmscfg) {
  auto dev = resolve_device(device);
  models::Yolo5Detect model(scale, nc);
  auto sd = serialization::load_state_dict(weights);
  int copied = model->load_from_state_dict(sd.entries);
  std::cout << "[v5-pred] loaded " << copied << " tensors from "
            << weights << "\n";
  model->to(dev); model->eval();

  cv::Mat src = cv::imread(in_path, cv::IMREAD_COLOR);
  if (src.empty()) throw std::runtime_error("could not read " + in_path);
  auto lb = letterbox(src, imgsz);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(dev);

  torch::Tensor pred;
  {
    torch::NoGradGuard ng;
    pred = model->forward_eval(x);
  }
  auto outs = nms(pred, nmscfg);
  if (outs[0].size(0) == 0) return {};
  auto det = outs[0].to(torch::kCPU);
  auto boxes = det.slice(1, 0, 4).contiguous();
  scale_boxes(boxes, lb);
  det.slice(1, 0, 4) = boxes;

  const auto& names = coco_names();
  std::vector<Detection> result;
  result.reserve(det.size(0));
  auto a = det.accessor<float, 2>();
  for (int i = 0; i < det.size(0); ++i) {
    Detection d;
    d.x1 = a[i][0]; d.y1 = a[i][1]; d.x2 = a[i][2]; d.y2 = a[i][3];
    d.conf = a[i][4]; d.cls = (int)a[i][5];
    result.push_back(d);
    cv::Scalar color((d.cls * 41) % 256, (d.cls * 73) % 256, (d.cls * 11) % 256);
    cv::rectangle(src, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2}, color, 2);
    std::string label = (d.cls >= 0 && d.cls < (int)names.size()
                             ? names[d.cls] : std::to_string(d.cls));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
    cv::putText(src, buf, {(int)d.x1 + 2, std::max(14, (int)d.y1) - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  cv::imwrite(out_path, src);
  return result;
}

std::vector<Detection> predict_v11_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz, const std::string& device,
    int nc, models::Yolo11Scale scale, NMSConfig nmscfg) {
  auto dev = resolve_device(device);
  models::Yolo11Detect model(scale, nc);
  auto sd = serialization::load_state_dict(weights);
  int copied = model->load_from_state_dict(sd.entries);
  std::cout << "[v11-pred] loaded " << copied << " tensors from "
            << weights << "\n";
  model->to(dev); model->eval();

  cv::Mat src = cv::imread(in_path, cv::IMREAD_COLOR);
  if (src.empty()) throw std::runtime_error("could not read " + in_path);
  auto lb = letterbox(src, imgsz);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(dev);

  torch::Tensor pred;
  {
    torch::NoGradGuard ng;
    pred = model->forward_eval(x);
  }
  auto outs = nms(pred, nmscfg);
  if (outs[0].size(0) == 0) return {};
  auto det = outs[0].to(torch::kCPU);
  auto boxes = det.slice(1, 0, 4).contiguous();
  scale_boxes(boxes, lb);
  det.slice(1, 0, 4) = boxes;

  const auto& names = coco_names();
  std::vector<Detection> result;
  result.reserve(det.size(0));
  auto a = det.accessor<float, 2>();
  for (int i = 0; i < det.size(0); ++i) {
    Detection d;
    d.x1 = a[i][0]; d.y1 = a[i][1]; d.x2 = a[i][2]; d.y2 = a[i][3];
    d.conf = a[i][4]; d.cls = (int)a[i][5];
    result.push_back(d);
    cv::Scalar color((d.cls * 41) % 256, (d.cls * 73) % 256, (d.cls * 11) % 256);
    cv::rectangle(src, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2}, color, 2);
    std::string label = (d.cls >= 0 && d.cls < (int)names.size()
                             ? names[d.cls] : std::to_string(d.cls));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
    int baseline = 0;
    auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int yt = std::max((int)d.y1, sz.height + 4);
    cv::rectangle(src, {(int)d.x1, yt - sz.height - 4},
                  {(int)d.x1 + sz.width + 4, yt}, color, cv::FILLED);
    cv::putText(src, buf, {(int)d.x1 + 2, yt - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  cv::imwrite(out_path, src);
  return result;
}

std::vector<Detection> predict_v26_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz, const std::string& device,
    int nc, models::Yolo26Scale scale, NMSConfig nmscfg) {
  auto dev = resolve_device(device);
  models::Yolo26Detect model(scale, nc);
  auto sd = serialization::load_state_dict(weights);
  int copied = model->load_from_state_dict(sd.entries);
  std::cout << "[v26-pred] loaded " << copied << " tensors from "
            << weights << "\n";
  model->to(dev); model->eval();

  cv::Mat src = cv::imread(in_path, cv::IMREAD_COLOR);
  if (src.empty()) throw std::runtime_error("could not read " + in_path);
  auto lb = letterbox(src, imgsz);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(dev);

  torch::Tensor pred;
  {
    torch::NoGradGuard ng;
    pred = model->forward_eval(x);
  }
  auto outs = nms(pred, nmscfg);
  if (outs[0].size(0) == 0) return {};
  auto det = outs[0].to(torch::kCPU);
  auto boxes = det.slice(1, 0, 4).contiguous();
  scale_boxes(boxes, lb);
  det.slice(1, 0, 4) = boxes;

  const auto& names = coco_names();
  std::vector<Detection> result;
  result.reserve(det.size(0));
  auto a = det.accessor<float, 2>();
  for (int i = 0; i < det.size(0); ++i) {
    Detection d;
    d.x1 = a[i][0]; d.y1 = a[i][1]; d.x2 = a[i][2]; d.y2 = a[i][3];
    d.conf = a[i][4]; d.cls = (int)a[i][5];
    result.push_back(d);
    cv::Scalar color((d.cls * 41) % 256, (d.cls * 73) % 256, (d.cls * 11) % 256);
    cv::rectangle(src, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2}, color, 2);
    std::string label = (d.cls >= 0 && d.cls < (int)names.size()
                             ? names[d.cls] : std::to_string(d.cls));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s %.2f", label.c_str(), d.conf);
    int baseline = 0;
    auto sz = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int yt = std::max((int)d.y1, sz.height + 4);
    cv::rectangle(src, {(int)d.x1, yt - sz.height - 4},
                  {(int)d.x1 + sz.width + 4, yt}, color, cv::FILLED);
    cv::putText(src, buf, {(int)d.x1 + 2, yt - 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
  }
  cv::imwrite(out_path, src);
  return result;
}

}  // namespace yolocpp::inference
