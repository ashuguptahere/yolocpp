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
  out.pad_x = pad_x;
  out.pad_y = pad_y;

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
  // BGR uint8 HWC → BGR uint8 CHW. No dtype promotion, no normalisation.
  // The trainer does both on GPU after HtoD for the gpu_aug path.
  // Channel order stays BGR (matches what the existing GPU HSV helper
  // expects). Caller can `permute` back to HWC if needed.
  TORCH_CHECK(bgr.type() == CV_8UC3, "image_to_tensor_u8: need CV_8UC3 input");
  auto t = torch::from_blob(
               bgr.data, {bgr.rows, bgr.cols, 3},
               torch::TensorOptions().dtype(torch::kUInt8))
               .permute({2, 0, 1})
               .contiguous();
  return t.clone();  // own the data
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

}  // namespace yolocpp::inference
