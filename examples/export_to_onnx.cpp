// examples/export_to_onnx.cpp — export a `.pt` to ONNX or TensorRT.
//
// Build:  cmake --build build --target export_to_onnx
// Run:    ./build/examples/export_to_onnx yolo11s.pt out.onnx onnx
//         ./build/examples/export_to_onnx yolo11s.pt out.trt  trt

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <weights.pt> <out_path> [format=onnx|trt] [precision=fp16|fp32]\n";
    return 1;
  }
  const std::string format    = (argc >= 4) ? argv[3] : "onnx";
  const std::string precision = (argc >= 5) ? argv[4] : "fp16";

  yolocpp::YOLO model(argv[1]);
  model.export_({
      .format    = format,
      .out       = argv[2],
      .precision = precision,
  });
  std::cout << "[export_to_onnx] wrote " << argv[2]
            << " (" << format << "/" << precision << ")\n";
  return 0;
}
