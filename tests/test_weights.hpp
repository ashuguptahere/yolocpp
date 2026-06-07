#pragma once
// Shared test helper: locate a cached weight file by basename.
//
// Weights cache under ./models (the resolver's download dir as of the
// models/ convention); ./data is kept as a legacy fallback. Returns ""
// when neither directory has the file, so weight-gated smokes can SKIP
// (a pass) instead of aborting on a missing checkpoint — matching the
// "SKIP when weights/data missing" convention used across the suite.

#include <filesystem>
#include <string>

namespace yolocpp::test {

inline std::string find_weight(const std::string& basename) {
  namespace fs = std::filesystem;
  for (const char* dir : {"models/", "data/"}) {
    fs::path p = std::string(dir) + basename;
    if (fs::exists(p)) return p.string();
  }
  return {};
}

}  // namespace yolocpp::test
