// Dev-only utility: synthesise a small test video at /tmp/bus.mp4 by
// repeating data/bus.jpg N times. Used as a fixture for the
// frame-loop smoke test (#51C2). Not registered with ctest — built
// for manual invocation.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

int main(int argc, char** argv) {
  std::string src   = (argc > 1) ? argv[1] : "data/bus.jpg";
  std::string dst   = (argc > 2) ? argv[2] : "/tmp/bus.mp4";
  int        nframes = (argc > 3) ? std::stoi(argv[3]) : 30;
  double     fps    = (argc > 4) ? std::stod(argv[4]) : 10.0;

  auto img = cv::imread(src);
  if (img.empty()) {
    std::cerr << "[mktv] cannot read " << src << "\n";
    return 1;
  }
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  cv::VideoWriter w(dst, fourcc, fps, img.size());
  if (!w.isOpened()) {
    std::cerr << "[mktv] cannot open writer for " << dst << "\n";
    return 1;
  }
  for (int i = 0; i < nframes; ++i) w.write(img);
  std::cerr << "[mktv] wrote " << dst << " (" << nframes << " frames @ "
            << fps << " fps)\n";
  return 0;
}
