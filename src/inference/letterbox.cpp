#include "yolocpp/inference/letterbox.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

namespace yolocpp::inference {

LetterboxResult letterbox(const cv::Mat& src, int new_shape,
                          cv::Scalar pad_color, bool scale_up,
                          bool auto_minrec) {
  LetterboxResult out;
  out.orig_w = src.cols;
  out.orig_h = src.rows;

  // Scale ratio (gain).
  double r = std::min((double)new_shape / src.rows, (double)new_shape / src.cols);
  if (!scale_up) r = std::min(r, 1.0);

  int new_unpad_w = (int)std::round(src.cols * r);
  int new_unpad_h = (int)std::round(src.rows * r);

  int dw = new_shape - new_unpad_w;
  int dh = new_shape - new_unpad_h;
  if (auto_minrec) {  // make a multiple of 32
    dw %= 32;
    dh %= 32;
  }
  // split padding evenly
  double pad_x = dw / 2.0;
  double pad_y = dh / 2.0;
  out.gain  = r;

  cv::Mat resized;
  if (src.cols != new_unpad_w || src.rows != new_unpad_h) {
    cv::resize(src, resized, {new_unpad_w, new_unpad_h}, 0, 0, cv::INTER_LINEAR);
  } else {
    resized = src;
  }
  int top    = (int)std::round(pad_y - 0.1);
  int bottom = (int)std::round(pad_y + 0.1);
  int left   = (int)std::round(pad_x - 0.1);
  int right  = (int)std::round(pad_x + 0.1);
  // Store the ACTUALLY-APPLIED integer pad (the copyMakeBorder top/left), not
  // the fractional dw/2. scale_boxes / dataset label transforms invert this to
  // stay pixel-aligned with where the image content actually landed; the
  // fractional value shifted boxes by up to 0.5px on odd padding and diverged
  // from upstream scale_boxes (which uses round(dw/2 - 0.1)). val mAP is
  // invariant (GT placement + pred un-projection both use this same pad).
  out.pad_x = left;
  out.pad_y = top;
  cv::copyMakeBorder(resized, out.img, top, bottom, left, right,
                     cv::BORDER_CONSTANT, pad_color);
  return out;
}

torch::Tensor image_to_tensor(const cv::Mat& bgr) {
  // BGR uint8 HWC → RGB float32 CHW in [0, 1].
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat f;
  rgb.convertTo(f, CV_32F, 1.0 / 255.0);
  // HWC → CHW
  // f is HxWx3, contiguous. Build a tensor that owns its data.
  auto t = torch::from_blob(
               f.data, {f.rows, f.cols, 3},
               torch::TensorOptions().dtype(torch::kFloat32))
               .permute({2, 0, 1})
               .contiguous();
  return t.clone();  // detach from cv::Mat lifetime
}

torch::Tensor image_to_tensor_u8(const cv::Mat& bgr) {
  // BGR uint8 HWC → RGB uint8 CHW. No dtype promotion, no normalisation.
  // The trainer does both on GPU after HtoD for the gpu_aug path.
  // RGB output is essential: `image_to_tensor` (val path) returns RGB,
  // and Ultralytics pretrained weights are trained on RGB. If this
  // returns BGR, training operates on BGR-labelled-RGB pixels and the
  // model is silently retrained to expect a flipped channel order
  // vs the val pipeline — observed as a uniform ~0.10 mAP gap. The
  // GPU HSV helper's channel indices are now "labelled BGR" but
  // actually carry R/G/B order; the H/S/V formulas are still
  // mathematically valid color jitter (just slightly different
  // distribution than upstream's true RGB HSV jitter, which we accept
  // as a minor parity gap vs the major train/val color-channel gap
  // this fix closes).
  TORCH_CHECK(bgr.type() == CV_8UC3, "image_to_tensor_u8: need CV_8UC3 input");
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  // `.contiguous()` already allocates a fresh owning tensor when
  // the input is a strided view (which permute(2,0,1) of an HWC
  // tensor is). The previous `.clone()` after `.contiguous()` was
  // a redundant second memcpy — dropped, saves ~0.05 ms per call.
  // (#95 profile-guided.)
  return torch::from_blob(
             rgb.data, {rgb.rows, rgb.cols, 3},
             torch::TensorOptions().dtype(torch::kUInt8))
             .permute({2, 0, 1})
             .contiguous();
}

torch::Tensor image_to_tensor_u8_bgr_chw(const cv::Mat& bgr) {
  TORCH_CHECK(bgr.type() == CV_8UC3,
              "image_to_tensor_u8_bgr_chw: need CV_8UC3 input");
  // Skip cv::cvtColor entirely — alias bgr.data directly, permute
  // HWC→CHW, contiguous() copies into an owning buffer. Saves the
  // BGR→RGB cvtColor cost (~0.03 ms) and the intermediate cv::Mat
  // allocation. The caller is responsible for BGR→RGB on GPU
  // (cheap index_select on the channel dim).
  return torch::from_blob(
             bgr.data, {bgr.rows, bgr.cols, 3},
             torch::TensorOptions().dtype(torch::kUInt8))
             .permute({2, 0, 1})
             .contiguous();
}

void scale_boxes(torch::Tensor& xyxy, const LetterboxResult& lb) {
  // Inverse of letterbox: subtract pad, divide by gain, clamp to image.
  using torch::indexing::Slice;
  auto xs = xyxy.index({Slice(), Slice(0, 4, 2)});  // x1, x2
  auto ys = xyxy.index({Slice(), Slice(1, 4, 2)});  // y1, y2
  xs.sub_(lb.pad_x).div_(lb.gain);
  ys.sub_(lb.pad_y).div_(lb.gain);
  // Match the upstream clip_boxes: clip to [0, orig_w] / [0, orig_h]
  // (inclusive of the right/bottom edge), not [0, orig - 1].
  xs.clamp_(0, lb.orig_w);
  ys.clamp_(0, lb.orig_h);
}

GpuLetterboxBatch gpu_letterbox_batch(const std::vector<cv::Mat>& bgrs,
                                      int imgsz,
                                      torch::Device device) {
  namespace F = torch::nn::functional;
  GpuLetterboxBatch out;
  out.lbs.reserve(bgrs.size());
  std::vector<torch::Tensor> per_image;
  per_image.reserve(bgrs.size());

  // Per-image pipeline (#95C):
  //   1. H2D the raw HWC BGR uint8 bytes (no CPU resize, no cvtColor).
  //   2. Permute HWC → CHW, cast to float (kFloat), divide by 255.
  //   3. F::interpolate(bilinear, align_corners=false) to the
  //      aspect-ratio-preserving size.
  //   4. F::pad(constant=114/255) up to imgsz × imgsz.
  //   5. .flip(0) on the channel dim → BGR → RGB.
  // Each image's metadata (gain / pad_x / pad_y) is captured into a
  // LetterboxResult so scale_boxes() still works downstream.
  for (const auto& src : bgrs) {
    TORCH_CHECK(src.type() == CV_8UC3, "gpu_letterbox: need CV_8UC3");
    LetterboxResult lb;
    lb.orig_w = src.cols; lb.orig_h = src.rows;
    double r = std::min((double)imgsz / src.rows, (double)imgsz / src.cols);
    int new_w = (int)std::round(src.cols * r);
    int new_h = (int)std::round(src.rows * r);
    int dw = imgsz - new_w, dh = imgsz - new_h;
    double pad_x = dw / 2.0, pad_y = dh / 2.0;
    // Store the actually-applied integer pad (see letterbox() above) — the
    // F::pad() call below uses round(pad - 0.1) for the top/left, so
    // scale_boxes must invert that exact offset, not the fractional dw/2.
    lb.gain = r;
    lb.pad_x = std::round(pad_x - 0.1);
    lb.pad_y = std::round(pad_y - 0.1);

    // Upload raw HWC uint8 directly (cv::Mat data is HWC-contiguous, so this
    // is a clean contiguous H2D with NO CPU transpose), then do the HWC→CHW
    // permute + BGR→RGB flip + float cast + /255 entirely on the GPU. The old
    // path did a per-image CPU permute().contiguous() before the upload —
    // profiling showed that CPU transpose dominated gpu_letterbox.
    TORCH_CHECK(src.isContinuous(), "gpu_letterbox: need continuous cv::Mat");
    auto hwc_u8 = torch::from_blob(
                      src.data, {src.rows, src.cols, 3},
                      torch::TensorOptions().dtype(torch::kUInt8))
                      .to(device, /*non_blocking=*/true);      // [H, W, 3] BGR uint8 GPU
    auto chw    = hwc_u8.permute({2, 0, 1})                    // [3, H, W] BGR uint8 (view)
                       .flip(0)                                // BGR → RGB
                       .to(torch::kFloat)
                       .div_(255.0f)
                       .unsqueeze(0);                          // [1, 3, H, W] RGB float
    auto resized = F::interpolate(
        chw,
        F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(torch::kBilinear)
            .align_corners(false));
    int top    = (int)std::round(pad_y - 0.1);
    int bottom = (int)std::round(pad_y + 0.1);
    int left   = (int)std::round(pad_x - 0.1);
    int right  = (int)std::round(pad_x + 0.1);
    auto padded = F::pad(
        resized,
        F::PadFuncOptions({left, right, top, bottom})
            .mode(torch::kConstant)
            .value(114.0 / 255.0));               // pad colour, matches upstream
    per_image.push_back(padded.squeeze(0));       // [3, imgsz, imgsz]
    // For scale_boxes we keep `lb.img` empty — the GPU path doesn't
    // build a cv::Mat. Callers should use the returned tensor.
    out.lbs.push_back(std::move(lb));
  }
  out.x = torch::stack(per_image, /*dim=*/0).contiguous();
  return out;
}

}  // namespace yolocpp::inference
