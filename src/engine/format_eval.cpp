#include "yolocpp/engine/format_eval.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <torch/torch.h>

#include <memory>

#include "yolocpp/core/log.hpp"
#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"  // Detection

namespace yolocpp::engine {

metrics::mAPResult eval_predictor(const ImagePredictor& predict,
                                  const datasets::YoloDataset& ds, int nc,
                                  float conf, float iou) {
  std::vector<metrics::DetectionRow> dets;
  std::vector<metrics::GroundTruthRow> gts;
  inference::NMSConfig nm;
  nm.conf_thresh = conf;
  nm.iou_thresh  = iou;

  const std::size_t n = ds.size();
  for (std::size_t i = 0; i < n; ++i) {
    auto lb = ds.get_letterboxed_u8(i);              // imgsz BGR + pixel GT
    auto ds_dets = predict(lb.img, nm);
    for (const auto& d : ds_dets)
      dets.push_back({d.x1, d.y1, d.x2, d.y2, d.conf, d.cls, static_cast<int>(i)});
    // GT: flat (cls, cx, cy, w, h) in letterboxed-imgsz pixel coords.
    const auto& t = lb.targets;
    for (std::size_t k = 0; k + 4 < t.size(); k += 5) {
      float cx = t[k + 1], cy = t[k + 2], w = t[k + 3], h = t[k + 4];
      gts.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                     static_cast<int>(t[k]), static_cast<int>(i)});
    }
  }
  return metrics::compute_map(dets, gts, nc);
}

ImagePredictor make_onnx_predictor(const std::string& onnx_path, int imgsz, int nc) {
  auto net = std::make_shared<cv::dnn::Net>();
  try {
    *net = cv::dnn::readNetFromONNX(onnx_path);
    if (net->empty()) throw std::runtime_error("empty network");
    net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  } catch (const std::exception& e) {
    LOG_WARN("bench") << "ONNX not runnable by cv::dnn (" << e.what()
                      << ") — skipping ONNX mAP";
    return {};
  }
  // Probe forward: cv::dnn 4.6 now PARSES our graph (the DFL ReduceSum is
  // emitted axes-as-attribute, #70) but its forward can still fail on the
  // anchor/stride decode subgraph's shape handling. Run one dummy forward;
  // if it throws (or the output isn't the expected [1, 4+nc, A]), degrade to
  // "skipped" so the benchmark prints "-" rather than a misleading 0.000.
  // True ONNX-runtime mAP for the detect family is gated on adding
  // onnxruntime (a deliberately-avoided dep — maintainer decision, #70).
  try {
    cv::Mat probe = cv::dnn::blobFromImage(
        cv::Mat::zeros(imgsz, imgsz, CV_8UC3), 1.0 / 255.0,
        cv::Size(imgsz, imgsz), cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
    net->setInput(probe);
    cv::Mat po = net->forward();
    if (po.dims != 3)
      throw std::runtime_error("unexpected forward output dims=" +
                               std::to_string(po.dims));
  } catch (const std::exception& e) {
    LOG_WARN("bench")
        << "ONNX parses but cv::dnn can't run the decode graph (" << e.what()
        << ") — skipping ONNX mAP (needs onnxruntime; see #70)";
    return {};
  }

  return [net, imgsz, nc](const cv::Mat& bgr,
                          const inference::NMSConfig& nm) -> std::vector<inference::Detection> {
    std::vector<inference::Detection> out;
    try {
      cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0 / 255.0, cv::Size(imgsz, imgsz),
                                            cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
      net->setInput(blob);
      cv::Mat o = net->forward();                    // [1, 4+nc, A], CPU float32
      if (o.dims != 3) return out;
      auto pred = torch::from_blob(o.ptr<float>(),
                                   {o.size[0], o.size[1], o.size[2]}, torch::kFloat32)
                      .clone();
      auto per_image = inference::nms(pred, nm);     // vector<[N,6]>
      if (per_image.empty()) return out;
      auto d = per_image[0].to(torch::kCPU);
      auto a = d.accessor<float, 2>();
      for (int r = 0; r < d.size(0); ++r)
        out.push_back({a[r][0], a[r][1], a[r][2], a[r][3], a[r][4],
                       static_cast<int>(a[r][5])});
    } catch (const std::exception& e) {
      LOG_DEBUG("bench") << "cv::dnn forward failed: " << e.what();
    }
    return out;
  };
}

}  // namespace yolocpp::engine
