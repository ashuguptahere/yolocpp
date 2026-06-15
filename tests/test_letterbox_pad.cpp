// Regression test for the letterbox applied-pad bug.
//
// letterbox() pads the image via copyMakeBorder at integer offsets
// top/left = round(pad - 0.1), but used to STORE the fractional dw/2 in
// LetterboxResult.pad_x/pad_y. scale_boxes() and the dataset label transforms
// invert that stored pad, so on odd padding the stored fractional value was
// ~0.5px off from where the image content actually landed — shifting boxes and
// diverging from upstream scale_boxes (which subtracts round(dw/2 - 0.1)).
//
// We pick a size with ODD vertical padding so fractional != applied, then
// assert the stored pad equals the applied integer top/left (verified against
// the actual content offset in the padded image) and that a box round-trips
// exactly through (place at applied offset) → scale_boxes.

#include <torch/torch.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>

#include "yolocpp/inference/letterbox.hpp"

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; }  \
  } while (0)

int main() {
  using namespace yolocpp::inference;
  const int imgsz = 640;

  // 1000x752 → gain 0.64, new_unpad_h = round(752*0.64)=481 → dh=159 (odd),
  // pad_y fractional = 79.5, applied top = round(79.5-0.1)=79.
  const int W = 1000, H = 752;
  const cv::Scalar content(10, 20, 30), pad(114, 114, 114);
  cv::Mat src(H, W, CV_8UC3, content);

  auto lb = letterbox(src, imgsz, pad, /*scale_up=*/true, /*auto_minrec=*/false);

  // Recompute the applied integer offsets independently.
  double r = std::min((double)imgsz / H, (double)imgsz / W);
  int new_h = (int)std::round(H * r), new_w = (int)std::round(W * r);
  int top  = (int)std::round((imgsz - new_h) / 2.0 - 0.1);
  int left = (int)std::round((imgsz - new_w) / 2.0 - 0.1);
  // Precondition: vertical padding is odd, so the fractional pad (dh/2) is a
  // half-integer that differs from the applied integer top — exactly the case
  // the old code got wrong.
  EXPECT((imgsz - new_h) % 2 != 0,
         "test precondition: vertical padding is odd (fractional != applied)");
  EXPECT((double)top != (imgsz - new_h) / 2.0,
         "test precondition: applied top != fractional pad");

  // Stored pad must be the applied integer offset.
  EXPECT(lb.pad_y == std::floor(lb.pad_y), "pad_y is integral");
  EXPECT(lb.pad_x == std::floor(lb.pad_x), "pad_x is integral");
  EXPECT((int)lb.pad_y == top,  "pad_y == applied top offset");
  EXPECT((int)lb.pad_x == left, "pad_x == applied left offset");

  // Verify against the ACTUAL content placement in the padded image: the row
  // just above `top` is pad colour, row `top` is content (sampled mid-width).
  int xc = imgsz / 2;
  auto is_content = [&](int y) {
    auto px = lb.img.at<cv::Vec3b>(y, xc);
    return px[0] == 10 && px[1] == 20 && px[2] == 30;
  };
  EXPECT(top == 0 || !is_content(top - 1), "row above top offset is padding");
  EXPECT(is_content(top), "row at top offset is image content");

  // 2) A box round-trips exactly: place at original coords via the APPLIED
  //    offset, then scale_boxes back. Off by ~0.5px under the old fractional pad.
  float ox1 = 100, oy1 = 150, ox2 = 400, oy2 = 600;
  auto box = torch::tensor({{(float)(ox1 * lb.gain + left),
                             (float)(oy1 * lb.gain + top),
                             (float)(ox2 * lb.gain + left),
                             (float)(oy2 * lb.gain + top)}});
  scale_boxes(box, lb);
  auto a = box.accessor<float, 2>();
  double err = std::max({std::abs(a[0][0] - ox1), std::abs(a[0][1] - oy1),
                         std::abs(a[0][2] - ox2), std::abs(a[0][3] - oy2)});
  std::cerr << "[letterbox-pad] round-trip max err = " << err << " px\n";
  EXPECT(err < 1e-3, "box round-trips through scale_boxes within 1e-3 px");

  std::cout << "ALL PASS (pad_x=" << lb.pad_x << " pad_y=" << lb.pad_y << ")\n";
  return 0;
}
