#include "yolocpp/losses/yolo2_loss.hpp"

#include <algorithm>
#include <cmath>

namespace yolocpp::losses {

LossOutput Yolo2Loss::operator()(const std::vector<torch::Tensor>& feats,
                                  const torch::Tensor& tgt,
                                  const std::vector<double>& strides,
                                  int /*imgsz*/) const {
  // Targets arrive in post-letterbox PIXEL coords (the dataset
  // pipeline transforms YOLO-normalized → pixel inside `make_example`).
  // Convert into cell-units by dividing by the single stride.
  TORCH_CHECK(!strides.empty(),
              "Yolo2Loss: strides must have ≥1 entry (stride = imgsz/grid)");
  const float stride = (float)strides[0];
  TORCH_CHECK(feats.size() == 1,
              "Yolo2Loss expects 1 feature tensor, got ", feats.size());
  auto raw = feats[0];               // [B, na·(5+nc), H, W]
  const int B  = (int)raw.size(0);
  const int H  = (int)raw.size(2);
  const int W  = (int)raw.size(3);
  const int nc = cfg.nc;
  const int na = (int)cfg.anchors.size() / 2;
  TORCH_CHECK(raw.size(1) == na * (5 + nc),
              "Yolo2Loss: feature channels (", raw.size(1),
              ") != na·(5+nc) (", na * (5 + nc), ")");
  const auto opts = raw.options();
  auto device     = raw.device();

  // Reshape to [B, na, 5+nc, H, W] → [B, na, H, W, 5+nc].
  auto y = raw.view({B, na, 5 + nc, H, W}).permute({0, 1, 3, 4, 2}).contiguous();

  auto tx_raw = y.select(-1, 0);                    // [B, na, H, W]
  auto ty_raw = y.select(-1, 1);
  auto tw_raw = y.select(-1, 2);
  auto th_raw = y.select(-1, 3);
  auto to_raw = y.select(-1, 4);
  auto tc_raw = y.slice(-1, 5, 5 + nc);             // [B, na, H, W, nc]

  // Anchor table in cell units, broadcastable to [1, na, 1, 1].
  auto anchor_t = torch::from_blob(
      const_cast<float*>(cfg.anchors.data()), {na, 2},
      torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .to(device);                   // [na, 2]
  auto aw = anchor_t.select(-1, 0).view({1, na, 1, 1});
  auto ah = anchor_t.select(-1, 1).view({1, na, 1, 1});

  // Build per-cell-per-anchor targets via a CPU walk over the GT
  // tensor. Region matches are sparse (few GTs per image) so this
  // is cheap relative to the conv stack.
  auto obj_mask = torch::zeros({B, na, H, W}, opts);      // 1 where this (b, a, j, i) is responsible
  auto tx_tgt   = torch::zeros({B, na, H, W}, opts);      // sigmoid-target (in [0, 1], cell-local)
  auto ty_tgt   = torch::zeros({B, na, H, W}, opts);
  auto tw_tgt   = torch::zeros({B, na, H, W}, opts);      // log-target (raw t_w form)
  auto th_tgt   = torch::zeros({B, na, H, W}, opts);
  auto iou_tgt  = torch::zeros({B, na, H, W}, opts);      // IoU value for the responsible cell
  auto cls_tgt  = torch::full({B, na, H, W}, -1,
                                opts.dtype(torch::kLong)); // -1 = ignore (no GT here)

  if (tgt.numel() > 0) {
    auto cpu = tgt.detach().to(torch::kCPU).contiguous();
    auto a   = cpu.accessor<float, 2>();
    for (int64_t r = 0; r < cpu.size(0); ++r) {
      int bi  = (int)a[r][0];
      int c   = (int)a[r][1];
      float cx = a[r][2], cy = a[r][3], w = a[r][4], h = a[r][5];
      if (bi < 0 || bi >= B) continue;
      if (c < 0 || c >= nc)  continue;
      // Pixel-coord cx/cy/w/h → cell-unit coords (stride = imgsz/grid).
      float gx = cx / stride;
      float gy = cy / stride;
      int gi = std::clamp((int)std::floor(gx), 0, W - 1);
      int gj = std::clamp((int)std::floor(gy), 0, H - 1);
      float gtw_cells = w / stride;
      float gth_cells = h / stride;
      // Pick the best anchor by w/h IoU (both centered at the cell).
      int   best_a   = 0;
      float best_iou = -1.0f;
      for (int aIdx = 0; aIdx < na; ++aIdx) {
        float aw_ = cfg.anchors[aIdx * 2 + 0];
        float ah_ = cfg.anchors[aIdx * 2 + 1];
        float inter = std::min(gtw_cells, aw_) * std::min(gth_cells, ah_);
        float un    = gtw_cells * gth_cells + aw_ * ah_ - inter;
        float iou   = un > 0 ? inter / un : 0;
        if (iou > best_iou) { best_iou = iou; best_a = aIdx; }
      }
      obj_mask[bi][best_a][gj][gi] = 1.0f;
      tx_tgt  [bi][best_a][gj][gi] = gx - (float)gi;        // ∈ [0, 1)
      ty_tgt  [bi][best_a][gj][gi] = gy - (float)gj;
      tw_tgt  [bi][best_a][gj][gi] = std::log(std::max(1e-9f,
                                          gtw_cells / cfg.anchors[best_a * 2 + 0]));
      th_tgt  [bi][best_a][gj][gi] = std::log(std::max(1e-9f,
                                          gth_cells / cfg.anchors[best_a * 2 + 1]));
      iou_tgt [bi][best_a][gj][gi] = best_iou;
      cls_tgt [bi][best_a][gj][gi] = c;
    }
  }
  obj_mask = obj_mask.to(device);
  tx_tgt   = tx_tgt.to(device);
  ty_tgt   = ty_tgt.to(device);
  tw_tgt   = tw_tgt.to(device);
  th_tgt   = th_tgt.to(device);
  iou_tgt  = iou_tgt.to(device);
  cls_tgt  = cls_tgt.to(device);

  // Coord loss — only over matched (b, a, j, i).
  auto tx_sig = torch::sigmoid(tx_raw);
  auto ty_sig = torch::sigmoid(ty_raw);
  auto loss_xy = ((tx_sig - tx_tgt).pow(2) + (ty_sig - ty_tgt).pow(2)) * obj_mask;
  auto loss_wh = ((tw_raw - tw_tgt).pow(2) + (th_raw - th_tgt).pow(2)) * obj_mask;
  auto loss_coord = cfg.lambda_coord * (loss_xy + loss_wh).sum();

  // Objectness loss — matched cells target IoU; unmatched cells target 0.
  auto obj_sig = torch::sigmoid(to_raw);
  auto loss_obj   = cfg.lambda_obj   * ((obj_sig - iou_tgt).pow(2) * obj_mask).sum();
  auto loss_noobj = cfg.lambda_noobj * (obj_sig.pow(2) * (1.0f - obj_mask)).sum();

  // Class loss — softmax CE on matched cells. We use cross_entropy
  // which expects [N, C] logits + [N] targets; flatten matched cells
  // first.
  torch::Tensor loss_cls = torch::zeros({}, opts);
  auto matched = cls_tgt.ge(0).nonzero();           // [M, 4]
  if (matched.size(0) > 0) {
    auto bi = matched.select(1, 0);
    auto ai = matched.select(1, 1);
    auto ji = matched.select(1, 2);
    auto ii = matched.select(1, 3);
    auto logits  = tc_raw.index({bi, ai, ji, ii});  // [M, nc]
    auto targets = cls_tgt.index({bi, ai, ji, ii}); // [M]
    loss_cls = cfg.lambda_class *
               torch::nn::functional::cross_entropy(logits, targets,
                   torch::nn::functional::CrossEntropyFuncOptions()
                       .reduction(torch::kSum));
  }

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
