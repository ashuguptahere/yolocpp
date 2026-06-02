#include "yolocpp/inference/results.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace yolocpp::inference {

std::string Results::class_name(int c) const {
  if (c >= 0 && c < (int)names.size()) return names[(std::size_t)c];
  return std::to_string(c);
}

std::vector<std::array<float, 4>> Results::xyxy() const {
  std::vector<std::array<float, 4>> r;
  r.reserve(boxes.size());
  for (const auto& d : boxes) r.push_back({d.x1, d.y1, d.x2, d.y2});
  return r;
}

std::vector<std::array<float, 4>> Results::xyxyn() const {
  std::vector<std::array<float, 4>> r;
  r.reserve(boxes.size());
  const float w = orig_w > 0 ? (float)orig_w : 1.0f;
  const float h = orig_h > 0 ? (float)orig_h : 1.0f;
  for (const auto& d : boxes)
    r.push_back({d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h});
  return r;
}

std::vector<std::array<float, 4>> Results::xywh() const {
  std::vector<std::array<float, 4>> r;
  r.reserve(boxes.size());
  for (const auto& d : boxes) {
    float cx = 0.5f * (d.x1 + d.x2);
    float cy = 0.5f * (d.y1 + d.y2);
    float w  = d.x2 - d.x1;
    float h  = d.y2 - d.y1;
    r.push_back({cx, cy, w, h});
  }
  return r;
}

std::vector<std::array<float, 4>> Results::xywhn() const {
  auto xy = xywh();
  const float W = orig_w > 0 ? (float)orig_w : 1.0f;
  const float H = orig_h > 0 ? (float)orig_h : 1.0f;
  for (auto& a : xy) { a[0] /= W; a[1] /= H; a[2] /= W; a[3] /= H; }
  return xy;
}

std::vector<float> Results::conf() const {
  std::vector<float> r;
  r.reserve(boxes.size());
  for (const auto& d : boxes) r.push_back(d.conf);
  return r;
}

std::vector<int> Results::cls() const {
  std::vector<int> r;
  r.reserve(boxes.size());
  for (const auto& d : boxes) r.push_back(d.cls);
  return r;
}

cv::Mat Results::plot(int line_thickness, double font_scale) const {
  cv::Mat out;
  if (orig_img.empty()) return out;
  orig_img.copyTo(out);
  for (const auto& d : boxes) {
    cv::Scalar color{(double)((d.cls * 41) % 256),
                     (double)((d.cls * 73) % 256),
                     (double)((d.cls * 11) % 256)};
    cv::rectangle(out, {(int)d.x1, (int)d.y1}, {(int)d.x2, (int)d.y2},
                  color, line_thickness);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %.2f",
                  class_name(d.cls).c_str(), d.conf);
    cv::putText(out, buf, {(int)d.x1 + 2, (int)d.y1 + 14},
                cv::FONT_HERSHEY_SIMPLEX, font_scale,
                {255, 255, 255}, 1);
  }
  return out;
}

bool Results::save(const std::string& path) const {
  auto im = plot();
  if (im.empty()) return false;
  return cv::imwrite(path, im);
}

bool Results::save_txt(const std::string& path, bool normalised) const {
  std::ofstream f(path);
  if (!f) return false;
  auto coords = normalised ? xyxyn() : xyxy();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& d = boxes[i];
    const auto& c = coords[i];
    f << d.cls << ' ' << d.conf << ' '
      << c[0] << ' ' << c[1] << ' ' << c[2] << ' ' << c[3] << '\n';
  }
  return true;
}

namespace {
// Tiny escape helper for JSON string values — enough for class names.
std::string esc(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n";  break;
      default:   o += c;
    }
  }
  return o;
}
}  // namespace

std::string Results::json() const {
  std::ostringstream o;
  o << std::fixed << std::setprecision(3);
  o << "{\"boxes\":[";
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& d = boxes[i];
    if (i) o << ",";
    o << "{\"cls\":" << d.cls
      << ",\"name\":\"" << esc(class_name(d.cls)) << "\""
      << ",\"conf\":" << d.conf
      << ",\"xyxy\":[" << d.x1 << "," << d.y1 << ","
                       << d.x2 << "," << d.y2 << "]}";
  }
  o << "],\"orig_shape\":[" << orig_h << "," << orig_w << "]"
    << ",\"speed\":{\"preprocess\":" << speed.preprocess_ms
    << ",\"inference\":"    << speed.inference_ms
    << ",\"postprocess\":"  << speed.postprocess_ms << "}}";
  return o.str();
}

void Results::print(std::ostream& os) const {
  os << "Results: " << boxes.size() << " detections on "
     << orig_w << "x" << orig_h << " image\n";
  os << "  speed: preprocess=" << speed.preprocess_ms << "ms"
     << "  inference=" << speed.inference_ms << "ms"
     << "  postprocess=" << speed.postprocess_ms << "ms\n";
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& d = boxes[i];
    os << "  [" << i << "] cls=" << d.cls << " (" << class_name(d.cls) << ")"
       << " conf=" << std::fixed << std::setprecision(3) << d.conf
       << " xyxy=(" << d.x1 << "," << d.y1 << ","
                    << d.x2 << "," << d.y2 << ")\n";
  }
}

}  // namespace yolocpp::inference
