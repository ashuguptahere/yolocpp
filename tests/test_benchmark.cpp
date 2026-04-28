// Verify the benchmark runs all three backends and reports plausible
// numbers (median > 0, TRT faster than PT on the 5090).

#include <iostream>

#include "yolocpp/engine/benchmark.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  yolocpp::engine::BenchConfig cfg;
  cfg.weights      = "data/yolov8n.pt";
  cfg.source       = "data/bus.jpg";
  cfg.imgsz        = 640;
  cfg.warmup_iters = 5;
  cfg.iters        = 30;
  cfg.cache_dir    = "build/bench_cache";

  auto rows = yolocpp::engine::run_benchmark(cfg);
  yolocpp::engine::print_benchmark(rows);

  EXPECT(rows.size() == 3, "expected 3 backends (PT, TRT-FP32, TRT-FP16)");
  for (const auto& r : rows) {
    EXPECT(r.median_ms > 0.0, "median should be > 0 for " + r.backend);
    EXPECT(r.median_ms < 100.0,
           "median > 100 ms suspicious for " + r.backend);
    EXPECT(r.num_detections >= 4, "expected ≥4 detections for " + r.backend);
  }
  // PT should not be faster than TRT-FP16 on a 5090.
  EXPECT(rows[0].median_ms >= rows[2].median_ms,
         "PT median should be ≥ TRT FP16 median");

  std::cout << "=== benchmark test PASS ===\n";
  return 0;
}
