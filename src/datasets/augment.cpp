#include "yolocpp/datasets/augment.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace yolocpp::datasets {

void hsv_jitter(cv::Mat& bgr, std::mt19937& rng,
                float hg, float sg, float vg) {
  // Matches ultralytics RandomHSV exactly (data/augment.py):
  //   r = uniform(-1, 1) * [hgain, sgain, vgain]
  //   lut_hue = ((x + r[0] * 180) % 180)         ← ADDITIVE + WRAP
  //   lut_sat = clip(x * (r[1] + 1), 0, 255)     ← multiplicative + clip
  //   lut_val = clip(x * (r[2] + 1), 0, 255)     ← multiplicative + clip
  //   lut_sat[0] = 0   (8.3.79+: keep pure white pure)
  //
  // The earlier per-loader form used `hue *= r_h` with a per-pixel
  // scalar wrap loop — that's multiplicative-non-wrapping, biases hues
  // toward 0/179 (red/violet), and is slower than this LUT-with-wrap.
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  const float r_h = u(rng) * hg;
  const float r_s = u(rng) * sg;
  const float r_v = u(rng) * vg;

  // Build the three LUTs to match upstream exactly.
  std::array<uint8_t, 256> lut_h{}, lut_s{}, lut_v{};
  const float hue_off = r_h * 180.f;
  for (int x = 0; x < 256; ++x) {
    // Hue: additive offset, wrap mod 180. `% 180` in numpy on a
    // negative integer returns non-negative; emulate that with
    // `((a % 180) + 180) % 180`.
    int hv = (int)std::lround((float)x + hue_off) % 180;
    if (hv < 0) hv += 180;
    lut_h[x] = (uint8_t)hv;
    // Sat / Val: multiplicative gain (r+1) with hard clip.
    int sv = (int)std::lround((float)x * (r_s + 1.f));
    int vv = (int)std::lround((float)x * (r_v + 1.f));
    lut_s[x] = (uint8_t)std::clamp(sv, 0, 255);
    lut_v[x] = (uint8_t)std::clamp(vv, 0, 255);
  }
  lut_s[0] = 0;  // 8.3.79+ rule — preserve pure white

  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> ch;
  cv::split(hsv, ch);
  cv::Mat lut_h_mat(1, 256, CV_8UC1, lut_h.data());
  cv::Mat lut_s_mat(1, 256, CV_8UC1, lut_s.data());
  cv::Mat lut_v_mat(1, 256, CV_8UC1, lut_v.data());
  cv::LUT(ch[0], lut_h_mat, ch[0]);
  cv::LUT(ch[1], lut_s_mat, ch[1]);
  cv::LUT(ch[2], lut_v_mat, ch[2]);
  cv::merge(ch, hsv);
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

}  // namespace yolocpp::datasets
