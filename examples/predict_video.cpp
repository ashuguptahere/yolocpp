// examples/predict_video.cpp — frame-loop inference (#51C2).
//
// Build:  cmake --build build --target predict_video
// Run:    ./build/examples/predict_video yolo11s.pt input.mp4 out.mp4
//         ./build/examples/predict_video yolo11s.pt rtsp://cam/stream
//         ./build/examples/predict_video yolo11s.pt 0          # webcam
//
// `--source` accepts a video file path, an HTTP/RTSP URL, or an
// integer webcam index. The API opens `cv::VideoCapture(source)`
// internally and writes annotated frames as an mp4 to `--out`
// (default: runs/predict/<stem>.mp4). Webcam runs cap at 600 frames.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <weights.pt> <video|url|webcam-idx> [out.mp4]\n";
    return 1;
  }
  yolocpp::YOLO model(argv[1]);
  model.to("auto");

  // Frame-loop predict returns an empty dets vector — per-frame dets
  // live in the on-disk mp4. A future enhancement could thread them
  // back as a vector<vector<Detection>>.
  model.predict({
      .source = argv[2],
      .out    = (argc >= 4) ? argv[3] : "",
  });
  std::cout << "[predict_video] done — see "
            << (argc >= 4 ? argv[3] : "runs/predict/<stem>.mp4")
            << " for the annotated output.\n";
  return 0;
}
