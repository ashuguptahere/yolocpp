// examples/train_finetune.cpp — fine-tune a pretrained model.
//
// Build:  cmake --build build --target train_finetune
// Run:    ./build/examples/train_finetune yolo11s.pt coco/data.yaml
//
// Loads the upstream pretrained `.pt` as the starting point and
// fine-tunes for 10 epochs on the supplied dataset, with
// deterministic seeding and post-train ONNX export.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <init_weights.pt> <data.yaml-or-dir>\n";
    return 1;
  }
  yolocpp::YOLO model(argv[1]);
  model.to("auto");

  model.train({
      .data    = argv[2],
      .imgsz   = 640,
      .epochs  = 10,
      .batch   = 16,
      .lr0     = 0.001,         // standard finetune LR
      .save    = "runs/train/finetune",
      .seed    = 42,
      .export_after_train = "onnx",  // writes runs/train/finetune/best.onnx
  });
  std::cout << "[train_finetune] done — see runs/train/finetune/.\n";
  return 0;
}
