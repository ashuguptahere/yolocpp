// examples/benchmark_model.cpp — latency / throughput benchmark.
//
// Build:  cmake --build build --target benchmark_model
// Run:    ./build/examples/benchmark_model yolo11s.pt bus.jpg
//
// Times PT (libtorch FP32) + TRT FP32 + TRT FP16 backends over
// `iters` runs and prints the median / p95 / mean ms + img/s.
// The TRT engine is built once and cached under `--cache`.

#include <iostream>
#include <yolocpp/api.hpp>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <weights.pt> <image>\n";
    return 1;
  }
  yolocpp::YOLO model(argv[1]);
  model.benchmark({
      .source = argv[2],
      .warmup = 10,
      .iters  = 100,
      .cache  = "build/bench_cache",
  });
  return 0;
}
