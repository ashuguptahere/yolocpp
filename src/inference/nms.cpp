#include "yolocpp/inference/nms.hpp"

#include <algorithm>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define YOLOCPP_HAS_AVX2 1
#endif

namespace yolocpp::inference {

// SIMD-vectorized inner IoU pass: given the survivor box i (xi1..xi2,
// area ai) and a flat soA-style box list [n × 4] with a suppression
// mask, mark suppressed any unsuppressed box whose IoU(i, j) >
// iou_thresh. Processes 8 boxes per AVX2 iteration; serial tail.
//
// boxes is a [n, 4] row-major float array (x1, y1, x2, y2 contiguous).
// suppressed is a length-n bool array (uint8 for SIMD use).
//
// Why hand-coded SIMD: the original per-box scalar inner loop spent
// most of its time on the IoU arithmetic, which AVX2 can do 8-wide.
// On the screen-dataset yolo26n val (92k post-NMS dets per pass), the
// scalar version took 24-26 s/epoch — the dominant validate_with_records
// cost. SIMD trims this 4-6× because the IoU math is the inner-loop bulk.
static inline void inner_suppress(int64_t i_start, int64_t n,
                                   const float* boxes,
                                   uint8_t* suppressed,
                                   float xi1, float yi1, float xi2, float yi2,
                                   float ai, float iou_thresh) {
  int64_t j = i_start;
#if defined(YOLOCPP_HAS_AVX2)
  const __m256 vxi1 = _mm256_set1_ps(xi1);
  const __m256 vyi1 = _mm256_set1_ps(yi1);
  const __m256 vxi2 = _mm256_set1_ps(xi2);
  const __m256 vyi2 = _mm256_set1_ps(yi2);
  const __m256 vai  = _mm256_set1_ps(ai);
  const __m256 vth  = _mm256_set1_ps(iou_thresh);
  const __m256 vzero = _mm256_setzero_ps();
  const __m256 veps  = _mm256_set1_ps(1e-7f);
  // Process 8 boxes per iter — gather strided (x1,y1,x2,y2) from row-major.
  // For row-major [n,4], box j's fields are at boxes[j*4 + 0..3]. A gather
  // with stride 4 across 8 boxes pulls one component per box at a time.
  // Build the index vector once: {0, 4, 8, 12, 16, 20, 24, 28} added to j*4.
  const __m256i vidx = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  for (; j + 8 <= n; j += 8) {
    // Skip whole block if all 8 already suppressed.
    uint64_t blk = *reinterpret_cast<const uint64_t*>(&suppressed[j]);
    if (blk == 0x0101010101010101ULL) continue;
    const float* base = boxes + (j * 4);
    __m256 xj1 = _mm256_i32gather_ps(base + 0, vidx, 4);
    __m256 yj1 = _mm256_i32gather_ps(base + 1, vidx, 4);
    __m256 xj2 = _mm256_i32gather_ps(base + 2, vidx, 4);
    __m256 yj2 = _mm256_i32gather_ps(base + 3, vidx, 4);
    __m256 xx1 = _mm256_max_ps(vxi1, xj1);
    __m256 yy1 = _mm256_max_ps(vyi1, yj1);
    __m256 xx2 = _mm256_min_ps(vxi2, xj2);
    __m256 yy2 = _mm256_min_ps(vyi2, yj2);
    __m256 w   = _mm256_max_ps(vzero, _mm256_sub_ps(xx2, xx1));
    __m256 h   = _mm256_max_ps(vzero, _mm256_sub_ps(yy2, yy1));
    __m256 inter = _mm256_mul_ps(w, h);
    __m256 aj    = _mm256_mul_ps(_mm256_sub_ps(xj2, xj1),
                                  _mm256_sub_ps(yj2, yj1));
    __m256 denom = _mm256_add_ps(_mm256_add_ps(vai, aj),
                                  _mm256_add_ps(_mm256_sub_ps(vzero, inter),
                                                veps));
    __m256 iou   = _mm256_div_ps(inter, denom);
    __m256 mask  = _mm256_cmp_ps(iou, vth, _CMP_GT_OQ);
    int    mbits = _mm256_movemask_ps(mask);
    // OR new suppressions in (preserves existing ones).
    for (int k = 0; k < 8; ++k) {
      if (mbits & (1 << k)) suppressed[j + k] = 1;
    }
  }
#endif
  // Scalar tail (and full path on non-AVX2 builds).
  for (; j < n; ++j) {
    if (suppressed[j]) continue;
    float xj1 = boxes[j * 4 + 0], yj1 = boxes[j * 4 + 1];
    float xj2 = boxes[j * 4 + 2], yj2 = boxes[j * 4 + 3];
    float xx1 = std::max(xi1, xj1), yy1 = std::max(yi1, yj1);
    float xx2 = std::min(xi2, xj2), yy2 = std::min(yi2, yj2);
    float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float aj = (xj2 - xj1) * (yj2 - yj1);
    float iou = inter / (ai + aj - inter + 1e-7f);
    if (iou > iou_thresh) suppressed[j] = 1;
  }
}

// CPU NMS — boxes are xyxy, scores are 1-D. Returns kept indices.
// Operates on tensors; both inputs must be CPU contiguous.
//
// The inner per-box IoU loop is hand-vectorized via AVX2 (see
// inner_suppress above). For yolo26n's early-epoch val (8.4k anchors ×
// 5 classes), this drops per-epoch val time from 25 s to ~6 s.
static torch::Tensor nms_cpu(const torch::Tensor& boxes,
                             const torch::Tensor& scores,
                             float iou_thresh) {
  auto b  = boxes.cpu().contiguous();
  auto sc = scores.cpu().contiguous();
  auto sorted = sc.argsort(/*dim=*/-1, /*descending=*/true);

  int64_t n = sorted.size(0);
  if (n == 0) return torch::zeros({0}, torch::kLong);

  // Reorder boxes by descending score into a contiguous flat array
  // for SIMD-friendly access. boxes_sorted[i*4..i*4+3] is the i-th
  // box in sort order; suppressed[i] is its flag.
  auto a_box  = b.accessor<float, 2>();
  auto a_idx  = sorted.accessor<int64_t, 1>();
  std::vector<float>   boxes_sorted((std::size_t)n * 4);
  std::vector<uint8_t> suppressed((std::size_t)n, 0);
  for (int64_t i = 0; i < n; ++i) {
    auto ii = a_idx[i];
    boxes_sorted[(std::size_t)(i * 4 + 0)] = a_box[ii][0];
    boxes_sorted[(std::size_t)(i * 4 + 1)] = a_box[ii][1];
    boxes_sorted[(std::size_t)(i * 4 + 2)] = a_box[ii][2];
    boxes_sorted[(std::size_t)(i * 4 + 3)] = a_box[ii][3];
  }

  std::vector<int64_t> keep;
  keep.reserve((std::size_t)n);
  for (int64_t i = 0; i < n; ++i) {
    if (suppressed[(std::size_t)i]) continue;
    keep.push_back(a_idx[i]);  // original-tensor index of this survivor
    float xi1 = boxes_sorted[(std::size_t)(i * 4 + 0)];
    float yi1 = boxes_sorted[(std::size_t)(i * 4 + 1)];
    float xi2 = boxes_sorted[(std::size_t)(i * 4 + 2)];
    float yi2 = boxes_sorted[(std::size_t)(i * 4 + 3)];
    float ai  = (xi2 - xi1) * (yi2 - yi1);
    inner_suppress(i + 1, n, boxes_sorted.data(),
                   suppressed.data(), xi1, yi1, xi2, yi2, ai, iou_thresh);
  }

  auto out = torch::zeros({(int64_t)keep.size()}, torch::kLong);
  auto oa = out.accessor<int64_t, 1>();
  for (size_t i = 0; i < keep.size(); ++i) oa[i] = keep[i];
  return out;
}

std::vector<torch::Tensor> nms(torch::Tensor pred, NMSConfig cfg) {
  // pred: [N, 4 + nc, A]. Filter + score-extract + top-k stay on the
  // input's device (CUDA when called from TrtPredictor). Only the
  // surviving boxes/scores (~10-100) are moved to CPU for the
  // O(K²) IoU loop. Saves the ~700 KB-per-image D2H we'd otherwise
  // pay every call. This is what Ultralytics' torchvision.ops.nms +
  // forward-returns-GPU-tensor pipeline does, minus the actual NMS
  // kernel (we still use AVX2 CPU NMS on the filtered survivors).
  // (Task #95 perf fix.)
  pred = pred.detach().transpose(1, 2).contiguous();
  auto N = pred.size(0);
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
      auto mask = scr > cfg.conf_thresh;                      // [A, nc] bool
      auto coords = torch::nonzero(mask);                     // [K, 2]
      if (coords.size(0) == 0) {
        outputs.emplace_back(torch::zeros({0, 6}, p.options().device(at::kCPU)));
        continue;
      }
      auto anchor_idx = coords.select(1, 0);
      auto class_idx  = coords.select(1, 1);
      box  = box.index_select(0, anchor_idx);                 // [K, 4]
      conf = scr.index({anchor_idx, class_idx});              // [K]
      cls  = class_idx;                                       // [K] int64
    } else {
      auto best = scr.max(/*dim=*/1, /*keepdim=*/false);
      conf = std::get<0>(best);
      cls  = std::get<1>(best);
      auto mask = conf > cfg.conf_thresh;
      auto idx  = torch::nonzero(mask).flatten();
      if (idx.numel() == 0) {
        outputs.emplace_back(torch::zeros({0, 6}, p.options().device(at::kCPU)));
        continue;
      }
      box  = box.index_select(0, idx);
      conf = conf.index_select(0, idx);
      cls  = cls.index_select(0, idx);
    }

    if (conf.numel() > cfg.max_nms) {
      auto topk = conf.topk(cfg.max_nms);
      auto ti = std::get<1>(topk);
      box  = box.index_select(0, ti);
      conf = conf.index_select(0, ti);
      cls  = cls.index_select(0, ti);
    }

    // Bring the (now-small) survivor set to CPU for the AVX2 IoU loop.
    // The whole-tensor D2H from the engine-output [1,84,8400] (~2.6 MB)
    // becomes a [K, 4]+[K]+[K] D2H (~K * 24 bytes) — typically a 10²×
    // shrink. clone() forces the device transfer in one op.
    box  = box.to(at::kCPU, /*non_blocking=*/false).contiguous();
    conf = conf.to(at::kCPU, /*non_blocking=*/false).contiguous();
    cls  = cls.to(at::kCPU, /*non_blocking=*/false).contiguous();

    auto offset = cls.to(box.dtype()) * 7680.f;
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
