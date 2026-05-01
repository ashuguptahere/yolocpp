#include "yolocpp/inference/nms.hpp"

#include <algorithm>
#include <vector>

namespace yolocpp::inference {

// CPU NMS — boxes are xyxy, scores are 1-D. Returns kept indices.
// Operates on tensors; both inputs must be CPU contiguous.
static torch::Tensor nms_cpu(const torch::Tensor& boxes,
                             const torch::Tensor& scores,
                             float iou_thresh) {
  auto b  = boxes.cpu().contiguous();
  auto sc = scores.cpu().contiguous();
  auto sorted = sc.argsort(/*dim=*/-1, /*descending=*/true);
  auto a_box = b.accessor<float, 2>();
  auto a_idx = sorted.accessor<int64_t, 1>();

  int64_t n = sorted.size(0);
  std::vector<bool> suppressed(n, false);
  std::vector<int64_t> keep;
  keep.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    auto ii = a_idx[i];
    if (suppressed[i]) continue;
    keep.push_back(ii);
    float xi1 = a_box[ii][0], yi1 = a_box[ii][1],
          xi2 = a_box[ii][2], yi2 = a_box[ii][3];
    float ai = (xi2 - xi1) * (yi2 - yi1);
    for (int64_t j = i + 1; j < n; ++j) {
      if (suppressed[j]) continue;
      auto jj = a_idx[j];
      float xj1 = a_box[jj][0], yj1 = a_box[jj][1],
            xj2 = a_box[jj][2], yj2 = a_box[jj][3];
      float xx1 = std::max(xi1, xj1), yy1 = std::max(yi1, yj1);
      float xx2 = std::min(xi2, xj2), yy2 = std::min(yi2, yj2);
      float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
      float inter = w * h;
      float aj = (xj2 - xj1) * (yj2 - yj1);
      float iou = inter / (ai + aj - inter + 1e-7f);
      if (iou > iou_thresh) suppressed[j] = true;
    }
  }
  auto out = torch::zeros({(int64_t)keep.size()}, torch::kLong);
  auto oa = out.accessor<int64_t, 1>();
  for (size_t i = 0; i < keep.size(); ++i) oa[i] = keep[i];
  return out;
}

std::vector<torch::Tensor> nms(torch::Tensor pred, NMSConfig cfg) {
  // pred: [N, 4 + nc, A] → permute to [N, A, 4 + nc]; CPU only.
  pred = pred.detach().to(torch::kCPU).transpose(1, 2).contiguous();
  auto N = pred.size(0);
  auto A = pred.size(1);
  auto C = pred.size(2);
  int  nc = (int)C - 4;

  std::vector<torch::Tensor> outputs;
  outputs.reserve(N);

  for (int64_t b = 0; b < N; ++b) {
    auto p = pred[b];  // [A, 4 + nc]
    auto box = p.slice(/*dim=*/1, 0, 4);
    auto scr = p.slice(/*dim=*/1, 4, 4 + nc);

    torch::Tensor conf, cls;
    if (cfg.multi_label) {
      // For every (anchor, class) pair with score > conf, emit one row.
      // Matches the upstream val multi_label=True path.
      auto mask = scr > cfg.conf_thresh;                      // [A, nc] bool
      auto coords = torch::nonzero(mask);                     // [K, 2] (anchor, class)
      if (coords.size(0) == 0) {
        outputs.emplace_back(torch::zeros({0, 6}, p.options()));
        continue;
      }
      auto anchor_idx = coords.select(1, 0);
      auto class_idx  = coords.select(1, 1);
      box  = box.index_select(0, anchor_idx);                 // [K, 4]
      conf = scr.index({anchor_idx, class_idx});              // [K]
      cls  = class_idx;                                       // [K] int64
    } else {
      // Single-label: pick the best class per anchor.
      auto best = scr.max(/*dim=*/1, /*keepdim=*/false);
      conf = std::get<0>(best);  // [A]
      cls  = std::get<1>(best);  // [A] int64
      auto mask = conf > cfg.conf_thresh;
      auto idx  = torch::nonzero(mask).flatten();
      if (idx.numel() == 0) {
        outputs.emplace_back(torch::zeros({0, 6}, p.options()));
        continue;
      }
      box  = box.index_select(0, idx);
      conf = conf.index_select(0, idx);
      cls  = cls.index_select(0, idx);
    }

    if (conf.numel() > cfg.max_nms) {
      // keep top-max_nms by confidence
      auto topk = conf.topk(cfg.max_nms);
      auto ti = std::get<1>(topk);
      box  = box.index_select(0, ti);
      conf = conf.index_select(0, ti);
      cls  = cls.index_select(0, ti);
    }

    // Class-aware NMS via a per-class offset (standard upstream trick).
    auto offset = cls.to(box.dtype()) * 7680.f;  // larger than any image
    auto box_off = box.clone();
    box_off.select(1, 0).add_(offset);
    box_off.select(1, 1).add_(offset);
    box_off.select(1, 2).add_(offset);
    box_off.select(1, 3).add_(offset);

    auto keep = nms_cpu(box_off, conf, cfg.iou_thresh);
    if (keep.size(0) > cfg.max_det) keep = keep.slice(0, 0, cfg.max_det);

    auto out = torch::cat({box.index_select(0, keep),
                           conf.index_select(0, keep).unsqueeze(1),
                           cls.to(box.dtype()).index_select(0, keep).unsqueeze(1)},
                          /*dim=*/1);
    outputs.emplace_back(out);
  }
  return outputs;
}

}  // namespace yolocpp::inference
