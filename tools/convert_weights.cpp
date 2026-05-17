// One-shot tool: walk known cache / current-dir paths for legacy
// Darknet `.weights` binaries and convert each to a `.pt` state-dict
// in `data/`. Once this has run, the rest of the codebase consumes
// the `.pt` outputs and never re-reads the original `.weights`.
//
// The `src/serialization/{darknet,yolov1,yolov2}_weights.cpp`
// converters stay in the build (other users may want to run a
// conversion themselves) — this tool just drives them once.
//
// Usage:
//   build/convert_weights              # convert all locally-available .weights
//   build/convert_weights yolo4 yolo2  # convert specific entries
//
// Exit code: number of files that FAILED to convert (0 on full success).
//

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/serialization/darknet_weights.hpp"
#include "yolocpp/serialization/yolov1_weights.hpp"
#include "yolocpp/serialization/yolov2_weights.hpp"

namespace fs = std::filesystem;

namespace {

fs::path home_cache() {
  const char* h = std::getenv("HOME");
  return fs::path(h ? h : "/tmp") / ".cache/yolocpp/weights";
}

// Resolve `wname` against a fixed list of search roots, in order.
fs::path find_weights(const std::string& wname) {
  std::vector<fs::path> roots = {
      fs::current_path() / "data" / wname,
      fs::current_path() / wname,
      home_cache() / wname,
      fs::path("/tmp") / wname,
  };
  for (const auto& r : roots) {
    if (fs::exists(r) && fs::is_regular_file(r)) return r;
  }
  return {};
}

struct Entry {
  std::string id;                   // short id ("yolo4", "yolo2", ...)
  std::string weights_name;         // upstream filename
  std::string pt_name;              // our canonical .pt name (under data/)
  enum class Kind { V1, V2_Full, V2_Tiny, V4 } kind;
  int nc;
};

const std::vector<Entry>& kEntries() {
  static const std::vector<Entry> v = {
      // pjreddie / AlexeyAB Darknet binaries we know how to parse.
      {"yolo1",          "yolov1.weights",          "yolo1.pt",          Entry::Kind::V1,      20},
      {"yolo1-tiny",     "yolov1-tiny.weights",     "yolo1-tiny.pt",     Entry::Kind::V1,      20},
      {"yolo2",          "yolov2.weights",          "yolo2.pt",          Entry::Kind::V2_Full, 80},
      {"yolo2-voc",      "yolov2-voc.weights",      "yolo2-voc.pt",      Entry::Kind::V2_Full, 20},
      {"yolo2-tiny",     "yolov2-tiny.weights",     "yolo2-tiny.pt",     Entry::Kind::V2_Tiny, 80},
      {"yolo2-tiny-voc", "yolov2-tiny-voc.weights", "yolo2-tiny-voc.pt", Entry::Kind::V2_Tiny, 20},
      {"yolo4",          "yolov4.weights",          "yolo4.pt",          Entry::Kind::V4,      80},
  };
  return v;
}

bool convert_one(const Entry& e, const fs::path& data_dir) {
  fs::path src = find_weights(e.weights_name);
  if (src.empty()) {
    std::cout << "[skip] " << e.id << ": no " << e.weights_name
              << " found in data/, cwd, " << home_cache().string()
              << ", or /tmp\n";
    return true;  // skip, not failure
  }
  fs::path dst = data_dir / e.pt_name;
  if (fs::exists(dst)) {
    std::cout << "[ok  ] " << e.id << ": " << dst << " already exists\n";
    return true;
  }
  fs::create_directories(data_dir);
  std::cout << "[run ] " << e.id << ": " << src << " → " << dst << "\n";
  try {
    using namespace yolocpp::serialization;
    int n = 0;
    switch (e.kind) {
      case Entry::Kind::V1:
        n = convert_yolov1_weights(src.string(), dst.string(), e.nc);
        break;
      case Entry::Kind::V2_Full:
        n = convert_yolov2_weights(src.string(), dst.string(), e.nc,
                                    yolocpp::models::Yolo2Scale::Full);
        break;
      case Entry::Kind::V2_Tiny:
        n = convert_yolov2_weights(src.string(), dst.string(), e.nc,
                                    yolocpp::models::Yolo2Scale::Tiny);
        break;
      case Entry::Kind::V4:
        n = convert_yolov4_weights(src.string(), dst.string(), e.nc);
        break;
    }
    std::cout << "[done] " << e.id << ": " << n << " blocks → " << dst << "\n";
    return true;
  } catch (const std::exception& ex) {
    std::cerr << "[FAIL] " << e.id << ": " << ex.what() << "\n";
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  fs::path data_dir = fs::current_path() / "data";
  std::vector<std::string> want;
  for (int i = 1; i < argc; ++i) want.emplace_back(argv[i]);

  int failures = 0;
  for (const auto& e : kEntries()) {
    if (!want.empty() &&
        std::find(want.begin(), want.end(), e.id) == want.end()) {
      continue;
    }
    if (!convert_one(e, data_dir)) ++failures;
  }
  if (failures == 0) {
    std::cout << "[convert_weights] all done\n";
  } else {
    std::cerr << "[convert_weights] " << failures << " failure(s)\n";
  }
  return failures;
}
