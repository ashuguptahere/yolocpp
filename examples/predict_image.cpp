// examples/predict_image.cpp — single-image inference via the public API.
//
// Build:  cmake --build build --target predict_image
// Run:    ./build/examples/predict_image yolo11s.pt bus.jpg out.jpg
//
// This is the smallest possible yolocpp program. Loads weights,
// runs predict on one image, prints the detection count + per-det
// bbox/conf/class.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <weights.pt> <image> [out.jpg]\n";
    return 1;
  }
  const std::string weights = argv[1];
  const std::string source  = argv[2];
  const std::string out     = (argc >= 4) ? argv[3] : "";

  yolocpp::YOLO model(weights);
  model.to("auto");  // pick CUDA when available, else CPU

  auto dets = model.predict({
      .source = source,
      .out    = out,
      .conf   = 0.25f,
  });

  std::cout << "[predict_image] " << dets.size() << " detections\n";
  for (std::size_t i = 0; i < dets.size(); ++i) {
    const auto& d = dets[i];
    std::cout << "  [" << i << "] cls=" << d.cls
              << " conf=" << d.conf
              << " bbox=[" << d.x1 << ", " << d.y1 << ", "
              << d.x2 << ", " << d.y2 << "]\n";
  }
  return 0;
}
