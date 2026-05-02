// examples/predict_directory.cpp — fan out over a directory of images.
//
// Build:  cmake --build build --target predict_directory
// Run:    ./build/examples/predict_directory yolo11s.pt frames/ annotated/
//
// Every image in <input_dir> gets annotated and written to
// <output_dir> with a `_detect.jpg` suffix. For #52A3 we'll also
// surface a per-input map of detections; today the API returns the
// LAST image's dets only.

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
  // dispatcher which expands it to a sorted list of images and
  // writes one annotated jpg per input under `--out`.
  auto last_dets = model.predict({
      .source = argv[2],
      .out    = argv[3],
  });
  std::cout << "[predict_directory] last-image dets: " << last_dets.size()
            << "\n";
  std::cout << "[predict_directory] check " << argv[3]
            << "/ for per-image annotated jpgs.\n";
  return 0;
}
