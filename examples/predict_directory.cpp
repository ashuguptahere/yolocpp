// examples/predict_directory.cpp — fan out over a directory of images.
//
// Build:  cmake --build build --target predict_directory
// Run:    ./build/examples/predict_directory yolo11s.pt frames/ annotated/
//
// Every image in <input_dir> gets annotated and written to
// <output_dir> with a `_detect.jpg` suffix. `predict_many` (#52A3)
// returns the per-input detections keyed by image path, so you can
// post-process each image's boxes without re-running inference.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <weights.pt> <input_dir> <output_dir>\n";
    return 1;
  }

  yolocpp::YOLO model(argv[1]);
  model.to("auto");

  // Pass the directory as `--source`; the API delegates to the CLI
  // dispatcher which expands it to a sorted list of images, writes one
  // annotated jpg per input under `--out`, and (via predict_many)
  // returns {input_path, dets} for every image.
  auto per_image = model.predict_many({
      .source = argv[2],
      .out    = argv[3],
  });
  std::cout << "[predict_directory] " << per_image.size()
            << " image(s) processed:\n";
  for (const auto& [path, dets] : per_image) {
    std::cout << "  " << path << " -> " << dets.size() << " dets\n";
  }
  std::cout << "[predict_directory] annotated jpgs under " << argv[3] << "/\n";
  return 0;
}
