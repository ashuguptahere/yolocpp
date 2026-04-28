#include "yolocpp/cli/resolve.hpp"

#include "yolocpp/cli/data_yaml.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace yolocpp::cli {

namespace {

fs::path home_cache() {
  const char* home = std::getenv("HOME");
  fs::path p = (home ? home : "/tmp");
  return p / ".cache" / "yolocpp";
}

bool run_curl(const std::string& url, const fs::path& dst) {
  fs::create_directories(dst.parent_path());
  std::ostringstream cmd;
  cmd << "curl -fSL --retry 3 --retry-delay 2 -o '" << dst.string()
      << "' '" << url << "'";
  std::cerr << "[download] " << url << "\n";
  int rc = std::system(cmd.str().c_str());
  return rc == 0 && fs::exists(dst) && fs::file_size(dst) > 0;
}

bool run_unzip(const fs::path& zip, const fs::path& out_dir,
               const std::string& include_glob = "",
               bool flatten = false) {
  fs::create_directories(out_dir);
  std::ostringstream cmd;
  cmd << "unzip -qq -o";
  if (flatten) cmd << " -j";
  cmd << " '" << zip.string() << "' ";
  if (!include_glob.empty()) cmd << "'" << include_glob << "' ";
  cmd << " -d '" << out_dir.string() << "'";
  return std::system(cmd.str().c_str()) == 0;
}

// Where Ultralytics publishes most of their assets:
//   https://github.com/ultralytics/assets/releases/download/v8.3.0/<basename>
// And dataset zips:
//   https://github.com/ultralytics/assets/releases/download/v0.0.0/<name>.zip
const std::string kAssetBase = "https://github.com/ultralytics/assets/releases/download/v8.3.0";

// Ultralytics publishes weights at `yolov<N>...pt` upstream (e.g. yolov5n.pt,
// yolov8n.pt) and the new official lines `yolo11<x>.pt` / `yolo26<x>.pt`
// without the 'v'. We accept BOTH the upstream name and our canonical
// `yolo<N>` form. When the local file is missing we map back to the
// upstream URL via `upstream_basename()` below.
bool looks_like_ultralytics_weight(const std::string& base) {
  static const std::regex re(
      R"(^(yolo[0-9]+[nsmlxce]?u?(-cls|-seg|-pose|-obb)?|yolo3(-tiny|-spp)?u?|rtdetr-[lx])\.pt$)");
  return std::regex_match(base, re);
}

// Translate a canonical local name (yolo5n.pt, yolo8n-seg.pt, yolo11n.pt) to
// the upstream Ultralytics filename. v3/v5/v8/v9/v10 are published as
// `yolov<N>...pt`; v11/v26 are published without the 'v' (they're already
// canonical). Returns the input unchanged if no transform is needed.
std::string upstream_basename(const std::string& base) {
  static const std::regex re(R"(^yolo([0-9]+)(.*)$)");
  std::smatch m;
  if (!std::regex_match(base, m, re)) return base;
  std::string num  = m[1].str();
  std::string rest = m[2].str();
  // Ultralytics upstream uses "yolov<N>" for v3..v10 and "yolo<N>" for v11+.
  int n = std::stoi(num);
  if (n >= 11) return base;        // already canonical upstream
  return "yolov" + num + rest;     // re-insert the 'v'
}

}  // anonymous namespace

std::string resolve_weights(const std::string& spec) {
  // 1) If exists at the given path, use it.
  if (fs::exists(spec) && fs::is_regular_file(spec)) return spec;

  // 2) If it's a bare basename, try common locations.
  fs::path p(spec);
  std::string base = p.filename().string();
  if (base != spec) {
    // user gave a relative or absolute path; honour it (already missed step 1)
  }
  std::vector<fs::path> candidates = {
      fs::current_path() / "data" / base,
      fs::current_path() / base,
      home_cache() / "weights" / base,
  };
  for (const auto& c : candidates) {
    if (fs::exists(c) && fs::is_regular_file(c)) return c.string();
  }

  // 3) If recognized Ultralytics name, download.
  //    The local cache is keyed by our canonical `yolo<N>` name, but
  //    upstream still publishes v3..v10 as `yolov<N>...pt`, so we fetch
  //    from the v-prefixed URL and save under the canonical name.
  if (looks_like_ultralytics_weight(base)) {
    auto target = home_cache() / "weights" / base;
    std::string upstream = upstream_basename(base);
    if (run_curl(kAssetBase + "/" + upstream, target)) {
      std::cerr << "[resolve] cached " << base << " at " << target
                << " (upstream: " << upstream << ")\n";
      return target.string();
    }
  }

  throw std::runtime_error("could not resolve weights: " + spec +
                           " (checked cwd, ./data, ~/.cache/yolocpp/weights)");
}


// Ensure `dst` looks like a YOLO dataset (has `images/` or `train`/`val`).
// If empty/missing and `download_url` is set, fetch + unzip into `dst`'s
// parent (Ultralytics zips conventionally have a single top-level dir
// matching the dataset name). Returns true if the layout is in place.
namespace {
bool ensure_layout(const fs::path& dst, const std::string& download_url) {
  if (fs::exists(dst / "images") ||
      fs::exists(dst / "train") || fs::exists(dst / "val"))
    return true;
  if (download_url.empty()) return false;

  fs::path parent = dst.parent_path();
  fs::create_directories(parent);
  fs::path zip = parent / fs::path(download_url).filename();
  if (!fs::exists(zip) || fs::file_size(zip) == 0) {
    if (!run_curl(download_url, zip))
      throw std::runtime_error("failed to download dataset: " + download_url);
  }
  std::string ext = zip.extension().string();
  for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
  if (ext != ".zip")
    throw std::runtime_error("only .zip downloads are supported (got " +
                             zip.string() + ")");
  if (!run_unzip(zip, parent))
    throw std::runtime_error("failed to unzip: " + zip.string());

  return fs::exists(dst / "images") ||
         fs::exists(dst / "train") || fs::exists(dst / "val");
}
}  // namespace

// `data=` accepts ONLY a path to a YAML file (Ultralytics data.yaml form).
// Directories and bare names are rejected by design — the YAML is the
// single source of truth for `path`, `train`, `val`, `names`, `download`.
std::string resolve_dataset(const std::string& spec) {
  fs::path p(spec);
  std::string ext = p.extension().string();
  for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
  if (ext != ".yaml" && ext != ".yml")
    throw std::runtime_error(
        "data= must point at a .yaml/.yml file (got '" + spec +
        "'). Pass e.g. data=data/coco8/data.yaml");

  if (!fs::exists(spec))
    throw std::runtime_error("data yaml not found: " + spec);

  auto dy = parse_data_yaml(spec);
  fs::path dst = dy.root;
  if (!ensure_layout(dst, dy.download_url))
    throw std::runtime_error(
        "dataset root '" + dst.string() + "' has no YOLO layout "
        "(images/{train,val}). YAML did not provide a usable download: URL "
        "either.");
  return dst.string();
}

std::string scale_from_filename(const std::string& path) {
  fs::path p(path);
  std::string base = p.filename().string();
  // Match canonical "yolo<digits><scale>[u]?(-task)?.pt" — and also accept the
  // legacy upstream "yolov<digits>..." spelling for v3..v10 weights.
  std::smatch m;
  static const std::regex re(
      R"(yolov?[0-9]+([nsmlx])u?(?:-(?:cls|seg|pose|obb))?\.pt$)");
  if (std::regex_search(base, m, re)) return m[1].str();

  // Trained checkpoint (best.pt / last.pt) → look at sibling args.yaml's
  // `model:` line and re-extract from that filename.
  fs::path sibling = p.parent_path() / "args.yaml";
  if (fs::exists(sibling)) {
    std::ifstream f(sibling);
    std::string line;
    while (std::getline(f, line)) {
      auto pos = line.find("model:");
      if (pos == std::string::npos) continue;
      auto colon = line.find(':', pos);
      if (colon == std::string::npos) continue;
      std::string val = line.substr(colon + 1);
      while (!val.empty() && (val.front() == ' ' || val.front() == '"' ||
                              val.front() == '\t')) val.erase(0, 1);
      while (!val.empty() && (val.back() == ' ' || val.back() == '"' ||
                              val.back() == '\n' || val.back() == '\r' ||
                              val.back() == '\t')) val.pop_back();
      if (!val.empty()) {
        std::smatch m2;
        if (std::regex_search(val, m2, re)) return m2[1].str();
      }
    }
  }
  return "";
}

std::string version_from_filename(const std::string& path) {
  fs::path p(path);
  std::string base = p.filename().string();
  // Strip optional upstream "yolov" prefix → "yolo" so we only need one
  // table of canonical names below.
  if (base.rfind("yolov", 0) == 0) base = "yolo" + base.substr(5);

  // Order matters: longer prefixes first (yolo10/11/12/13/26 must be
  // matched before yolo1/yolo2 to avoid collapsing to a shorter version).
  if (base.rfind("yolo26", 0) == 0) return "v26";  // Ultralytics official
  if (base.rfind("yolo13", 0) == 0) return "v13";  // Lei et al. (unofficial)
  if (base.rfind("yolo12", 0) == 0) return "v12";  // Tian et al. (unofficial)
  if (base.rfind("yolo11", 0) == 0) return "v11";  // Ultralytics official
  if (base.rfind("yolo10", 0) == 0) return "v10";
  if (base.rfind("yolo9",  0) == 0) return "v9";
  if (base.rfind("yolo8",  0) == 0) return "v8";
  if (base.rfind("yolo7",  0) == 0) return "v7";
  if (base.rfind("yolo6",  0) == 0) return "v6";
  if (base.rfind("yolo5",  0) == 0) return "v5";
  if (base.rfind("yolo4",  0) == 0) return "v4";
  if (base.rfind("yolo3",  0) == 0) return "v3";
  if (base.rfind("rtdetr", 0) == 0) return "rtdetr";

  // Trained checkpoints (e.g. "best.pt") don't carry the version in their
  // filename. Try the sibling args.yaml: it has a `model: yolo5n.pt` line
  // that we can re-pattern-match on.
  fs::path sibling = p.parent_path() / "args.yaml";
  if (fs::exists(sibling)) {
    std::ifstream f(sibling);
    std::string line;
    while (std::getline(f, line)) {
      // Look for "model:" prefix.
      auto pos = line.find("model:");
      if (pos == std::string::npos) continue;
      auto colon = line.find(':', pos);
      if (colon == std::string::npos) continue;
      std::string val = line.substr(colon + 1);
      // Trim whitespace + quotes.
      while (!val.empty() && (val.front() == ' ' || val.front() == '"' ||
                              val.front() == '\t')) val.erase(0, 1);
      while (!val.empty() && (val.back() == ' ' || val.back() == '"' ||
                              val.back() == '\n' || val.back() == '\r' ||
                              val.back() == '\t')) val.pop_back();
      if (!val.empty()) {
        auto v = version_from_filename(val);
        // Avoid infinite recursion: if we got "v8" back from a default we
        // know nothing more, return that.
        return v;
      }
    }
  }
  return "v8";
}

}  // namespace yolocpp::cli
