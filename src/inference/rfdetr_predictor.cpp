// RF-DETR predict path (#65E). Letterbox → forward → decode (no NMS).

#include "yolocpp/inference/rfdetr_predictor.hpp"

#include <algorithm>

#include "yolocpp/inference/letterbox.hpp"

namespace yolocpp::inference {

std::vector<std::vector<Detection>> rfdetr_decode(const torch::Tensor& out,
                                                    int imgsz, float conf,
                                                    int max_det) {
  TORCH_CHECK(out.dim() == 3, "rfdetr_decode: expected [B, 4+nc, Q]");
  auto B  = out.size(0);
  auto Cn = out.size(1);
  auto Q  = out.size(2);
  auto nc = Cn - 4;
  TORCH_CHECK(nc > 0, "rfdetr_decode: bad channel count");

  // Move to CPU once.
  auto cpu = out.detach().to(torch::kCPU).contiguous();
  auto a   = cpu.accessor<float, 3>();

  std::vector<std::vector<Detection>> result(B);
  for (int64_t b = 0; b < B; ++b) {
    // Per-query best class.
    struct Cand {
      float score;
      int   cls;
      int   q;
    };
    std::vector<Cand> cands;
    cands.reserve(Q);
    for (int64_t q = 0; q < Q; ++q) {
      float best = -1.0f;
      int   bid  = 0;
      for (int64_t c = 0; c < nc; ++c) {
        float s = a[b][4 + c][q];
        if (s > best) { best = s; bid = static_cast<int>(c); }
      }
      if (best >= conf) cands.push_back({best, bid, static_cast<int>(q)});
    }

    // Top-K by score.
    if (max_det > 0 &&
        static_cast<int>(cands.size()) > max_det) {
      std::partial_sort(
          cands.begin(), cands.begin() + max_det, cands.end(),
          [](const Cand& a, const Cand& b) { return a.score > b.score; });
      cands.resize(max_det);
    } else {
      std::sort(cands.begin(), cands.end(),
                 [](const Cand& a, const Cand& b) { return a.score > b.score; });
    }

    auto& dets = result[b];
    dets.reserve(cands.size());
    for (auto& c : cands) {
      // forward_eval emits xyxy in pixel coords already (YOLO
      // contract); imgsz unused for box conversion but kept for API
      // symmetry with `Predictor::predict`.
      (void)imgsz;
      Detection d;
      d.x1 = a[b][0][c.q];
      d.y1 = a[b][1][c.q];
      d.x2 = a[b][2][c.q];
      d.y2 = a[b][3][c.q];
      d.conf = c.score;
      d.cls  = c.cls;
      dets.push_back(d);
    }
  }
  return result;
}

std::vector<Detection> rfdetr_predict_image(
    yolocpp::models::RFDetr& m, const cv::Mat& bgr, int imgsz,
    const torch::Device& device, float conf, int max_det) {
  auto lb = letterbox(bgr, imgsz);
  auto x  = image_to_tensor(lb.img).unsqueeze(0).to(device);

  torch::NoGradGuard ng;
  m->eval();
  auto out = m->forward_eval(x);

  auto per_image = rfdetr_decode(out, imgsz, conf, max_det);
  if (per_image.empty()) return {};
  auto& dets = per_image[0];

  // Unscale via the existing scale_boxes helper.
  if (!dets.empty()) {
    auto box_t = torch::empty({static_cast<int64_t>(dets.size()), 4},
                               torch::dtype(torch::kFloat));
    auto acc = box_t.accessor<float, 2>();
    for (size_t i = 0; i < dets.size(); ++i) {
      acc[i][0] = dets[i].x1; acc[i][1] = dets[i].y1;
      acc[i][2] = dets[i].x2; acc[i][3] = dets[i].y2;
    }
    scale_boxes(box_t, lb);
    auto acc2 = box_t.accessor<float, 2>();
    for (size_t i = 0; i < dets.size(); ++i) {
      dets[i].x1 = acc2[i][0]; dets[i].y1 = acc2[i][1];
      dets[i].x2 = acc2[i][2]; dets[i].y2 = acc2[i][3];
    }
  }
  return dets;
}

}  // namespace yolocpp::inference
