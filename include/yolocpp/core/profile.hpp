#pragma once
//
// Per-phase wall-clock profiler. Threaded throughout the inference
// + training hot paths so the user can answer "where did the cycles
// go?" without rewiring instrumentation each time. (Task #98.)
//
// Usage from CLI:
//   yolocpp --mode predict -m yolo11n.pt -s bus.jpg --profile
//   yolocpp --mode benchmark -m yolo11x.pt --profile
//   yolocpp --mode train ... --profile     (per-epoch breakdown)
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
  }
  ~ProfileScope() { Profile::instance().stop(tag_); }
 private:
  std::string tag_;
};

#define PROFILE_SCOPE(tag) ::yolocpp::core::ProfileScope _prof_scope_##__LINE__{tag}

}  // namespace yolocpp::core
