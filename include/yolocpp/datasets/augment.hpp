#pragma once
//
// Shared dataset augmentation primitives (#54E). Extracted so the four
// loaders (YoloDataset / FlatDataset / CocoDataset / VocDataset) share a
// single, parity-correct implementation instead of each carrying its own
// copy.

#include <random>

#include <opencv2/core.hpp>

namespace yolocpp::datasets {

// Upstream-parity HSV jitter (Ultralytics RandomHSV, data/augment.py):
//   r = uniform(-1, 1) * [hg, sg, vg]
//   hue: additive offset r[0]*180, wrapped mod 180   (256-entry cv::LUT)
//   sat/val: multiplicative gain (r+1) with hard clip (256-entry cv::LUT)
//   lut_sat[0] = 0   (8.3.79+: keep pure white pure)
// In-place on `bgr` (CV_8UC3). Uses cv::LUT — one vectorized table pass per
// channel, no per-pixel scalar loop.
void hsv_jitter(cv::Mat& bgr, std::mt19937& rng, float hg, float sg, float vg);

}  // namespace yolocpp::datasets
