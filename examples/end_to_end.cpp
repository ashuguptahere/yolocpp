// examples/end_to_end.cpp — full train → val → export → predict pipeline
// expressed in a single chained call sequence.
//
// Build:  cmake --build build --target end_to_end
// Run:    ./build/examples/end_to_end yolo11s.pt coco/data.yaml bus.jpg
//
// Demonstrates the chainable fluent API: `to()` and `task()` set
// per-instance defaults that subsequent calls inherit. `train()`,
// `export_()`, and `benchmark()` return `YOLO&` so calls can chain.
// `predict()` returns the detections so the caller can post-process
// or tally them.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <init.pt> <data.yaml-or-dir> <test_image>\n";
    return 1;
  }
  yolocpp::YOLO model(argv[1]);
  model
      .to("auto")
      .task("detect")
      .train({
          .data    = argv[2],
          .epochs  = 5,
          .batch   = 16,
          .save    = "runs/train/e2e",
          .seed    = 42,
          .export_after_train = "onnx",
      });

  // After training, runs/train/e2e/best.pt is the new checkpoint.
  // Use a fresh YOLO bound to it for val + predict.
  yolocpp::YOLO finetuned("runs/train/e2e/best.pt");
  finetuned.to("auto");

  finetuned.val({.data = argv[2]});
  auto dets = finetuned.predict({
      .source = argv[3],
      .out    = "runs/predict/e2e_test.jpg",
  });
  std::cout << "[end_to_end] post-train predict: " << dets.size()
            << " dets on " << argv[3] << "\n";
  return 0;
}
