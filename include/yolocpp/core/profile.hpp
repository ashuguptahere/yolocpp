#pragma once
//
// Per-phase wall-clock profiler. Each ProfileScope also emits an NVTX
// range when --profile is on, so `nsys profile ./yolocpp ... --profile`
// yields an Nsight Systems timeline for free. (Task #98 / #6.)
//
// Coverage (where ProfileScope is wired):
//   • benchmark mode (engine) — full preprocess/enqueue/nms breakdown.
//   • TRT predict path (trt_predictor.cpp) — gpu_letterbox / enqueueV3.
//   • C++ API Predictor::predict (predictor.cpp) — preprocess/forward/
//     nms/postprocess.
// Two paths are profiled differently and do NOT feed this singleton:
//   • the per-version CLI `.pt` predict functions report timing via
//     Results.speed (preprocess/inference/postprocess ms), not --profile.
//   • the training loop has per-step phase timers behind the
//     YOLOCPP_PROFILE_STEP env var (trainer.cpp), which need explicit
//     CUDA-stream syncs around each phase.
//
// Usage from CLI:
//   yolocpp --mode benchmark -m yolo11x.pt -s bus.jpg --profile
//   nsys profile -o yolo ./build/yolocpp --mode benchmark ... --profile
//
// Usage from C++ code:
//   {
//     PROFILE_SCOPE("letterbox");
//     auto lb = letterbox(bgr, imgsz);
//   }
// or:
//   Profile::start("h2d");
//   // ...
//   Profile::stop("h2d");
//
// All recorded times accumulate into a per-phase aggregator; a single
// `Profile::print_summary()` call at process exit emits a table with
// median + p95 + count per phase. RAII PROFILE_SCOPE handles the
// start/stop pair automatically. When disabled (the default), every
// method is a no-op — zero overhead.

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Optional NVTX3 ranges for nsys/Nsight Systems timelines. Header-only
// (ships with the CUDA toolkit, already on the include path via
// CUDA::cudart); compiled out cleanly if the header is ever absent.
#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define YOLOCPP_HAS_NVTX 1
#  endif
#endif

namespace yolocpp::core {

class Profile {
 public:
  static Profile& instance() {
    static Profile p;
    return p;
  }

  // Globally enable/disable. Off by default; the CLI flips it on when
  // --profile is passed.
  void set_enabled(bool on) { enabled_ = on; }
  bool enabled() const { return enabled_; }

  void start(const std::string& tag) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lk(mu_);
    starts_[tag] = std::chrono::steady_clock::now();
  }

  void stop(const std::string& tag) {
    if (!enabled_) return;
    auto t1 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    auto it = starts_.find(tag);
    if (it == starts_.end()) return;
    double ms = std::chrono::duration<double, std::milli>(t1 - it->second).count();
    samples_[tag].push_back(ms);
    starts_.erase(it);
  }

  void clear() {
    std::lock_guard<std::mutex> lk(mu_);
    starts_.clear();
    samples_.clear();
  }

  void print_summary(std::ostream& os) const;

 private:
  Profile() = default;
  bool enabled_ = false;
  mutable std::mutex mu_;
  std::map<std::string, std::chrono::steady_clock::time_point> starts_;
  std::map<std::string, std::vector<double>> samples_;
};

// RAII scope macro: PROFILE_SCOPE("h2d");  starts on construction,
// stops on destruction. Free when --profile is off.
class ProfileScope {
 public:
  explicit ProfileScope(std::string tag) : tag_(std::move(tag)) {
    Profile::instance().start(tag_);
#ifdef YOLOCPP_HAS_NVTX
    if (Profile::instance().enabled()) nvtxRangePushA(tag_.c_str());
#endif
  }
  ~ProfileScope() {
#ifdef YOLOCPP_HAS_NVTX
    if (Profile::instance().enabled()) nvtxRangePop();
#endif
    Profile::instance().stop(tag_);
  }
 private:
  std::string tag_;
};

#define PROFILE_SCOPE(tag) ::yolocpp::core::ProfileScope _prof_scope_##__LINE__{tag}

}  // namespace yolocpp::core
