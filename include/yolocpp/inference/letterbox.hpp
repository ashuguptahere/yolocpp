#pragma once
//
// Letterbox preprocessing — matches the upstream behaviour bit-for-bit.
//
// Resizes and pads an image so the long side equals `new_shape` while
// preserving aspect ratio, then returns the new image and the (gain, pad_x,
// pad_y) needed to map detections back to the original image coordinates.
//

#include <opencv2/core.hpp>
#include <torch/torch.h>

namespace yolocpp::inference {

struct LetterboxResult {
  cv::Mat   img;        // padded image (BGR, 8-bit, new_shape × new_shape)
  double    gain;       // scaling factor (new = old * gain)
  double    pad_x;      // x padding (left)
  double    pad_y;      // y padding (top)
  int       orig_w;
  int       orig_h;
};

LetterboxResult letterbox(const cv::Mat& src,
                          int   new_shape  = 640,
                          cv::Scalar pad_color = {114, 114, 114},
                          bool  scale_up    = true,
                          bool  auto_minrec = false);

// Convert a letterboxed BGR image to a CHW float32 tensor in [0, 1] on CPU.
torch::Tensor image_to_tensor(const cv::Mat& bgr);

// Map (x1, y1, x2, y2) boxes from letterboxed-image coordinates back to
// original-image coordinates. Operates in place on a [N, 4] float tensor.
void scale_boxes(torch::Tensor& xyxy, const LetterboxResult& lb);

}  // namespace yolocpp::inference
