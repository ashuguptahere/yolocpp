#include "yolocpp/losses/yolo1_loss.hpp"

#include <algorithm>
#include <cmath>

namespace yolocpp::losses {

namespace {

// IoU between two boxes in (cx, cy, w, h) form, all on the same scale.
// Returns scalar; safe with zero-area boxes (returns 0).
torch::Tensor iou_cxcywh(const torch::Tensor& a, const torch::Tensor& b) {
  auto ax1 = a.select(-1, 0) - a.select(-1, 2) * 0.5f;
  auto ay1 = a.select(-1, 1) - a.select(-1, 3) * 0.5f;
  auto ax2 = a.select(-1, 0) + a.select(-1, 2) * 0.5f;
  auto ay2 = a.select(-1, 1) + a.select(-1, 3) * 0.5f;
  auto bx1 = b.select(-1, 0) - b.select(-1, 2) * 0.5f;
  auto by1 = b.select(-1, 1) - b.select(-1, 3) * 0.5f;
  auto bx2 = b.select(-1, 0) + b.select(-1, 2) * 0.5f;
  auto by2 = b.select(-1, 1) + b.select(-1, 3) * 0.5f;
  auto inter_x1 = torch::max(ax1, bx1);
  auto inter_y1 = torch::max(ay1, by1);
  auto inter_x2 = torch::min(ax2, bx2);
  auto inter_y2 = torch::min(ay2, by2);
  auto iw = (inter_x2 - inter_x1).clamp_min(0);
  auto ih = (inter_y2 - inter_y1).clamp_min(0);
  auto inter = iw * ih;
  auto area_a = (ax2 - ax1).clamp_min(0) * (ay2 - ay1).clamp_min(0);
  auto area_b = (bx2 - bx1).clamp_min(0) * (by2 - by1).clamp_min(0);
  return inter / (area_a + area_b - inter + 1e-9);
}

}  // namespace

LossOutput Yolo1Loss::operator()(const std::vector<torch::Tensor>& feats,
                                  const torch::Tensor& tgt,
                                  const std::vector<double>& strides,
                                  int imgsz) const {
  // Targets arrive in post-letterbox PIXEL coords. Divide by `imgsz`
  // to recover the [0, 1] normalized form v1 uses internally
  // (stride = imgsz/S, so cx_norm = cx_pixel / (S·stride) = cx_pixel/imgsz).
  TORCH_CHECK(imgsz > 0, "Yolo1Loss: imgsz must be > 0");
  (void)strides;
  const float inv_imgsz = 1.0f / (float)imgsz;
  TORCH_CHECK(feats.size() == 1,
              "Yolo1Loss expects 1 feature tensor, got ", feats.size());
  // Compute the v1 loss in fp32 even under bf16 autocast: the CPU
  // accessor<float> loop below hard-requires float, and SSE accumulation
  // is steadier in fp32. The cast is autograd-safe (grads flow back to
  // the bf16 params) and a no-op when the output is already fp32.
  auto flat = feats[0].to(torch::kFloat);   // [B, S·S·(B·5+nc)]
  const int B   = (int)flat.size(0);
  const int S   = cfg.S;
  const int Bx  = cfg.B;
  const int nc  = cfg.nc;
  const int SS  = S * S;
  const auto opts = flat.options();
  auto device = flat.device();
  auto zero = torch::zeros({}, opts);

  // Split into the three Darknet contiguous blocks.
  auto cls   = flat.slice(1, 0, SS * nc).view({B, SS, nc});            // [B, SS, nc]
  auto conf  = flat.slice(1, SS * nc, SS * (nc + Bx)).view({B, SS, Bx});// [B, SS, Bx]
  auto coord = flat.slice(1, SS * (nc + Bx),
                            SS * (nc + Bx + 4 * Bx))
                    .view({B, SS, Bx, 4});                              // [B, SS, Bx, 4]

  // Mask of (b, cell) that contains any GT. We'll add to it as we
  // iterate through GTs below.
  auto cell_has_gt   = torch::zeros({B, SS}, opts.dtype(torch::kBool));
  auto target_cls    = torch::zeros({B, SS, nc}, opts);
  auto target_box    = torch::zeros({B, SS, 4}, opts);  // (tx, ty, sqrt_w, sqrt_h)
  // For each (b, cell) we'll mark exactly one "responsible" box index.
  // -1 means no GT.
  auto responsible_idx = torch::full({B, SS}, -1,
                                     opts.dtype(torch::kLong));
  auto target_iou    = torch::zeros({B, SS}, opts);  // IoU for the responsible box (confidence target)

  // Iterate through the flat target tensor and populate the per-cell
  // metadata. Most batches have a handful of GTs, so this CPU loop is
  // cheap relative to the forward pass.
  if (tgt.numel() > 0) {
    auto cpu_tgt = tgt.detach().to(torch::kCPU).contiguous();
    auto a = cpu_tgt.accessor<float, 2>();
    auto coord_cpu = coord.detach().to(torch::kCPU).contiguous();  // [B, SS, Bx, 4]
    auto coord_a = coord_cpu.accessor<float, 4>();
    for (int64_t r = 0; r < cpu_tgt.size(0); ++r) {
      int bi  = (int)a[r][0];
      int c   = (int)a[r][1];
      // Convert pixel-coord targets to normalized [0, 1].
      float cx = a[r][2] * inv_imgsz;
      float cy = a[r][3] * inv_imgsz;
      float w  = a[r][4] * inv_imgsz;
      float h  = a[r][5] * inv_imgsz;
      if (bi < 0 || bi >= B) continue;
      if (c < 0 || c >= nc)  continue;
      int gi = std::clamp((int)std::floor(cx * S), 0, S - 1);
      int gj = std::clamp((int)std::floor(cy * S), 0, S - 1);
      int cell = gj * S + gi;
      cell_has_gt[bi][cell] = true;
      target_cls[bi][cell][c] = 1.0f;
      // Box target: tx, ty (relative to cell) + sqrt(w), sqrt(h).
      float tx_gt = cx * S - (float)gi;
      float ty_gt = cy * S - (float)gj;
      target_box[bi][cell][0] = tx_gt;
      target_box[bi][cell][1] = ty_gt;
      target_box[bi][cell][2] = std::sqrt(std::max(0.f, w));
      target_box[bi][cell][3] = std::sqrt(std::max(0.f, h));
      // Pick the responsible box: highest IoU vs the GT.
      // GT in image-normalized cxcywh form: (cx, cy, w, h).
      float best_iou = -1.0f;
      int   best_b   = 0;
      for (int b = 0; b < Bx; ++b) {
        // Decode the predicted box at this cell, box-slot b.
        float tx = coord_a[bi][cell][b][0];
        float ty = coord_a[bi][cell][b][1];
        float pw = coord_a[bi][cell][b][2];
        float ph = coord_a[bi][cell][b][3];
        // Predicted box center in normalized [0,1] image coords:
        float pcx = ((float)gi + tx) / (float)S;
        float pcy = ((float)gj + ty) / (float)S;
        // v1 head emits sqrt(w), sqrt(h) — decode back.
        float bw = std::max(0.f, pw); bw = bw * bw;
        float bh = std::max(0.f, ph); bh = bh * bh;
        // IoU vs GT.
        float ax1 = pcx - bw * 0.5f, ay1 = pcy - bh * 0.5f;
        float ax2 = pcx + bw * 0.5f, ay2 = pcy + bh * 0.5f;
        float bx1 = cx  - w  * 0.5f, by1 = cy  - h  * 0.5f;
        float bx2 = cx  + w  * 0.5f, by2 = cy  + h  * 0.5f;
        float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
        float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
        float iw  = std::max(0.f, ix2 - ix1);
        float ih2 = std::max(0.f, iy2 - iy1);
        float inter = iw * ih2;
        float area_a = std::max(0.f, ax2 - ax1) * std::max(0.f, ay2 - ay1);
        float area_b = std::max(0.f, bx2 - bx1) * std::max(0.f, by2 - by1);
        float iou = inter / (area_a + area_b - inter + 1e-9f);
        if (iou > best_iou) { best_iou = iou; best_b = b; }
      }
      responsible_idx[bi][cell] = best_b;
      target_iou[bi][cell]      = best_iou;
    }
  }

  cell_has_gt    = cell_has_gt.to(device);
  target_cls     = target_cls.to(device);
  target_box     = target_box.to(device);
  responsible_idx = responsible_idx.to(device);
  target_iou     = target_iou.to(device);

  // Gather the responsible box's predictions per (b, cell). Where
  // cell_has_gt is false the index is -1 — clamp to 0 (the gathered
  // values are then masked out by cell_has_gt).
  auto safe_idx = responsible_idx.clamp_min(0).unsqueeze(-1);         // [B, SS, 1]
  auto resp_coord = coord.gather(2, safe_idx.unsqueeze(-1)
                                       .expand({B, SS, 1, 4}))
                          .squeeze(2);                                // [B, SS, 4]
  auto resp_conf  = conf.gather(2, safe_idx).squeeze(-1);             // [B, SS]

  auto mask = cell_has_gt.to(opts.dtype());                            // [B, SS]
  // Coordinate loss (only over responsible boxes).
  auto coord_diff = (resp_coord - target_box) * mask.unsqueeze(-1);
  auto loss_coord = cfg.lambda_coord * coord_diff.pow(2).sum();

  // Confidence loss — responsible boxes target IoU.
  auto loss_obj = ((resp_conf - target_iou) * mask).pow(2).sum();

  // No-obj loss — everything that is NOT the responsible box.
  //   * Non-responsible slots in cells that DO contain a GT
  //   * All slots in cells with no GT
  auto resp_onehot = torch::zeros({B, SS, Bx}, opts);
  resp_onehot.scatter_(2, safe_idx, 1.0f);                            // 1 at responsible slot
  resp_onehot = resp_onehot * mask.unsqueeze(-1);                     // zero out empty-cells
  auto noobj_mask = 1.0f - resp_onehot;                                // 1 everywhere except responsible
  auto loss_noobj = cfg.lambda_noobj * (conf.pow(2) * noobj_mask).sum();

  // Class loss — sum-squared per cell with a GT.
  auto loss_cls = ((cls - target_cls) * mask.unsqueeze(-1)).pow(2).sum();

  // Average over batch — keeps loss magnitude roughly stable across
  // different batch sizes.
  double batch_norm = std::max(1, B);
  auto total = (loss_coord + loss_obj + loss_noobj + loss_cls) / batch_norm;

  LossOutput o;
  o.total = total;
  o.box   = loss_coord.detach() / batch_norm;
  o.cls   = loss_cls.detach()   / batch_norm;
  o.dfl   = (loss_obj.detach() + loss_noobj.detach()) / batch_norm;
  return o;
}

}  // namespace yolocpp::losses
