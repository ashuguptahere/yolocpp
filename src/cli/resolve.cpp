#include "yolocpp/cli/resolve.hpp"

#include "yolocpp/cli/data_yaml.hpp"
#include "yolocpp/serialization/darknet_weights.hpp"
#include "yolocpp/serialization/yolov1_weights.hpp"
#include "yolocpp/serialization/yolov2_weights.hpp"
#include "yolocpp/serialization/yolov6_weights.hpp"
#include "yolocpp/serialization/yolov7_weights.hpp"
#include "yolocpp/serialization/yolov9_weights.hpp"
#include "yolocpp/serialization/yolov10_weights.hpp"
#include "yolocpp/serialization/yolov3_weights.hpp"

#include <algorithm>
#include <array>
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

// Canonical local model directory. The resolver searches it and lands all
// downloads here. Defaults to ./models (repo-relative); override with
// YOLOCPP_MODELS_DIR. Created on demand.
fs::path models_dir() {
  const char* env = std::getenv("YOLOCPP_MODELS_DIR");
  fs::path p = (env && env[0]) ? fs::path(env) : (fs::current_path() / "models");
  std::error_code ec;
  fs::create_directories(p, ec);
  return p;
}

// Wrap a string in single quotes for safe interpolation into a /bin/sh
// command, escaping any embedded single quote as '\'' . The URL and paths
// reaching run_curl/run_unzip are user-controlled (--dataset / --model), so
// an unescaped value like  http://x/'$(cmd)'.zip  would otherwise break out
// of the quotes and inject a shell command.
std::string sh_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

bool run_curl(const std::string& url, const fs::path& dst) {
  fs::create_directories(dst.parent_path());
  std::ostringstream cmd;
  cmd << "curl -fSL --retry 3 --retry-delay 2 -o " << sh_quote(dst.string())
      << ' ' << sh_quote(url);
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
  cmd << ' ' << sh_quote(zip.string()) << ' ';
  if (!include_glob.empty()) cmd << sh_quote(include_glob) << ' ';
  cmd << " -d " << sh_quote(out_dir.string());
  return std::system(cmd.str().c_str()) == 0;
}

// Upstream asset host for legacy YOLO releases (the only allow-listed
// mention of the upstream vendor in the codebase — see CLAUDE.md
// "remove every trace of upstream branding" / TODO #49). These are
// real network endpoints we can't rename; everything else in the
// codebase uses neutral terminology ("upstream", "the source `.pt`
// format").
//   weights : https://github.com/ultralytics/assets/releases/download/v8.3.0/<basename>
//   datasets: https://github.com/ultralytics/assets/releases/download/v0.0.0/<name>.zip
// Per-family upstreams that aren't Ultralytics: v6 ← meituan/YOLOv6,
// v7 ← WongKinYiu/yolov7, v13 ← iMoonLab/yolov13 (release tag `yolov13`).
const std::string kAssetBase = "https://github.com/ultralytics/assets/releases/download/v8.3.0";

// Upstream weights for v3/v5/v8/v9/v10 ship as `yolov<N>...pt` (note
// the 'v'); v11+ ship as `yolo<N>...pt` (no 'v') — already in our
// canonical form. We accept BOTH the upstream name and our canonical
// `yolo<N>` form. When the local file is missing we map back to the
// upstream URL via `upstream_basename()` below.
bool looks_like_upstream_weight(const std::string& base) {
  static const std::regex re(
      R"(^(yolo[0-9]+[nsmlxce]?u?(-cls|-seg|-pose|-obb)?|yolo3(-tiny|-spp)?u?)\.(pt|pth)$)");
  return std::regex_match(base, re);
}

// Translate a canonical local name (yolo5n.pt, yolo8n-seg.pt, yolo11n.pt)
// to the upstream filename. v3/v8/v9/v10 are published as
// `yolov<N>...pt`; v5 only as the anchor-free `yolov5<scale>u.pt`;
// v11+/v26 ship without the 'v' (already canonical). Returns the input
// unchanged if no transform is needed.
std::string upstream_basename(const std::string& base) {
  static const std::regex re(R"(^yolo([0-9]+)(.*)$)");
  std::smatch m;
  if (!std::regex_match(base, m, re)) return base;
  std::string num  = m[1].str();
  std::string rest = m[2].str();
  // Upstream uses "yolov<N>" for v3..v10 and "yolo<N>" for v11+.
  int n = std::stoi(num);
  if (n >= 11) return base;        // already canonical upstream
  if (n == 5) {
    // Ultralytics publishes v5 only as the anchor-free "u" form
    // (`yolov5<scale>u.pt`); our DFL/anchor-free v5 head loads it
    // directly. The classic anchor-based `yolov5<scale>.pt` is absent
    // from the assets release (404), so splice the 'u' in before the
    // extension.
    auto dot = rest.rfind('.');
    std::string stem = (dot == std::string::npos) ? rest : rest.substr(0, dot);
    std::string ext  = (dot == std::string::npos) ? "" : rest.substr(dot);
    return "yolov5" + stem + "u" + ext;
  }
  return "yolov" + num + rest;     // re-insert the 'v'
}

}  // anonymous namespace

std::string resolve_weights(const std::string& spec) {
  // 1) If exists at the given path, use it.
  if (fs::exists(spec) && fs::is_regular_file(spec)) return spec;

  // 2) If it's a bare basename, try common locations.
  fs::path p(spec);
  std::string base = p.filename().string();

  // 2a) Special-case yolo4.pt — convert from a Darknet `.weights` binary
  // if one is sitting next to it (or in the cache). This keeps the rest
  // of the predict path uniform: every model loads from a `.pt`.
  auto try_v4_convert = [&](const fs::path& pt_target) -> bool {
    static const std::vector<std::string> candidates = {
        "yolov4.weights", "yolo4.weights"};
    for (const auto& wname : candidates) {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / wname,
          fs::current_path() / wname,
          models_dir() / wname,
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov4_weights(w.string(), pt_target.string());
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
    }
    return false;
  };
  // 2b) Special-case Meituan YOLOv6 — convert upstream `.pt` to ours.
  // Supports yolo6{n,s,m,l}.pt + yolo6{s,m,l,x}_mbla.pt + yolo6{n,s,m,l}6.pt.
  // The same converter handles all twelve (RepConv fusion + `.block.` strip
  // with lookahead so RepBlock's / MBLABlock's ModuleList paths are preserved
  // + new rename rule for ERBlock_6 in P6 variants). MBLA scales use
  // ConvBNSiLU everywhere → no RepVGG branches to fuse. P6 variants use
  // the same RepVGG/BepC3 dispatch as standard but with one more stage.
  for (const std::string letter : {"n", "s", "m", "l",
                                    "s_mbla", "m_mbla", "l_mbla", "x_mbla",
                                    "n6", "s6", "m6", "l6"}) {
    std::string ours    = "yolo6"  + letter + ".pt";
    std::string upstream = "yolov6" + letter + ".pt";
    if (base != ours && base != upstream) continue;
    fs::path target = models_dir() / ours;
    if (fs::exists(target)) return target.string();
    auto try_v6_convert = [&](const fs::path& pt_target) -> bool {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / upstream,
          fs::current_path() / upstream,
          models_dir() / upstream,
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov6_pt(w.string(), pt_target.string());
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
      return false;
    };
    if (try_v6_convert(target)) return target.string();
    fs::path wsrc = models_dir() / upstream;
    const std::string url =
        "https://github.com/meituan/YOLOv6/releases/download/0.4.0/" + upstream;
    if (run_curl(url, wsrc) && try_v6_convert(target)) {
      return target.string();
    }
    break;
  }

  // 2c) Special-case YOLOv7 — convert from WongKinYiu's upstream `.pt`.
  // Supports yolo7.pt (base), yolo7-tiny.pt, yolo7x.pt. The same converter
  // handles all three (tiny has no RepConv blocks; the existing fusion
  // logic finds 3 RepConv pairs for base / x and 0 for tiny).
  for (auto pair : std::vector<std::pair<std::string, std::string>>{
           {"yolo7.pt",       "yolov7.pt"},
           {"yolo7-tiny.pt",  "yolov7-tiny.pt"},
           {"yolo7x.pt",      "yolov7x.pt"},
           {"yolo7-w6.pt",    "yolov7-w6.pt"},
           {"yolo7-e6.pt",    "yolov7-e6.pt"},
           {"yolo7-d6.pt",    "yolov7-d6.pt"},
           {"yolo7-e6e.pt",   "yolov7-e6e.pt"}}) {
    if (base != pair.first && base != pair.second) continue;
    fs::path target = models_dir() / pair.first;
    if (fs::exists(target)) return target.string();
    auto upstream = pair.second;
    auto try_v7_convert = [&](const fs::path& pt_target) -> bool {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / upstream,
          fs::current_path() / upstream,
          models_dir() / upstream,
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov7_pt(w.string(), pt_target.string());
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
      return false;
    };
    if (try_v7_convert(target)) return target.string();
    fs::path wsrc = models_dir() / upstream;
    const std::string url =
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/" + upstream;
    if (run_curl(url, wsrc) && try_v7_convert(target)) {
      return target.string();
    }
    break;
  }

  // 2d) Special-case YOLOv9 scales — convert the upstream `.pt`
  // to ours. Supports yolo9{t,s,m,c,e}.pt.
  for (const std::string letter : {"t", "s", "m", "c", "e"}) {
    std::string ours    = "yolo9"  + letter + ".pt";
    std::string upstream = "yolov9" + letter + ".pt";
    // Also accept the bare "yolo9.pt" → defaults to c.
    bool match = (base == ours || base == upstream)
                  || (letter == "c" && base == "yolo9.pt");
    if (!match) continue;
    fs::path target = models_dir() / ours;
    if (fs::exists(target)) return target.string();
    auto try_v9_convert = [&](const fs::path& pt_target) -> bool {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / upstream,
          fs::current_path() / upstream,
          models_dir() / upstream,
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov9_pt(w.string(), pt_target.string());
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
      return false;
    };
    if (try_v9_convert(target)) return target.string();
    fs::path wsrc = models_dir() / upstream;
    const std::string url =
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/" + upstream;
    if (run_curl(url, wsrc) && try_v9_convert(target)) {
      return target.string();
    }
    break;
  }

  // 2e) Special-case yolo10*.pt — convert from upstream yolov10{n,s,m,b,l,x}.pt
  // (one2many head dropped, RepVGGDW fusion, fp16→fp32). All 6 scales wired.
  // Letter detection: filename suffix before `.pt` (e.g. yolo10s.pt → "s").
  // The base "yolo10.pt" (no letter) is treated as the n scale for backwards
  // compat with the original single-scale naming.
  {
    auto v10_letter_from_base = [](const std::string& b) -> std::string {
      // Forms accepted: yolo10.pt, yolo10<L>.pt, yolov10<L>.pt where L ∈ {n,s,m,b,l,x}.
      if (b == "yolo10.pt") return "n";
      static const std::array<std::string, 6> letters = {"n", "s", "m", "b", "l", "x"};
      for (const auto& L : letters) {
        if (b == "yolo10" + L + ".pt")  return L;
        if (b == "yolov10" + L + ".pt") return L;
      }
      return "";
    };
    std::string letter = v10_letter_from_base(base);
    if (!letter.empty()) {
      // Target is yolo10<L>.pt in cache (or yolo10.pt for the historical n alias).
      std::string tgt_base =
          (base == "yolo10.pt") ? "yolo10.pt" : ("yolo10" + letter + ".pt");
      fs::path target = models_dir() / tgt_base;
      if (fs::exists(target)) return target.string();
      std::string upstream_base = "yolov10" + letter + ".pt";
      auto try_v10_convert = [&](const fs::path& pt_target) -> bool {
        std::vector<fs::path> roots = {
            fs::current_path() / "data" / upstream_base,
            fs::current_path() / upstream_base,
            models_dir() / upstream_base,
        };
        for (const auto& w : roots) {
          if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
          std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
          fs::create_directories(pt_target.parent_path());
          try {
            serialization::convert_yolov10_pt(w.string(), pt_target.string());
            return true;
          } catch (const std::exception& e) {
            std::cerr << "[resolve] convert failed: " << e.what() << "\n";
            return false;
          }
        }
        return false;
      };
      if (try_v10_convert(target)) return target.string();
      fs::path wsrc = models_dir() / upstream_base;
      const std::string url =
          "https://github.com/ultralytics/assets/releases/download/v8.3.0/" +
          upstream_base;
      if (run_curl(url, wsrc) && try_v10_convert(target)) {
        return target.string();
      }
    }
  }

  // 2e-bis) yolo13{n,s,l,x}.pt — iMoonLab fork (no `m` variant upstream).
  // Ships in deploy form and loads directly (no reparam/convert needed); just
  // fetch from the iMoonLab release if there's no local copy. Filenames: ours
  // `yolo13<L>.pt`, upstream `yolov13<L>.pt`.
  {
    static const std::array<std::string, 4> v13_letters = {"n", "s", "l", "x"};
    std::string letter;
    for (const auto& L : v13_letters) {
      if (base == "yolo13" + L + ".pt" || base == "yolov13" + L + ".pt") {
        letter = L;
        break;
      }
    }
    if (!letter.empty()) {
      std::string our_base      = "yolo13"  + letter + ".pt";
      std::string upstream_base = "yolov13" + letter + ".pt";
      // Prefer any local copy (data/, cwd, cache) under either name.
      for (const fs::path& c : {fs::current_path() / "data" / our_base,
                                fs::current_path() / "data" / upstream_base,
                                fs::current_path() / our_base,
                                models_dir() / our_base}) {
        if (fs::exists(c) && fs::is_regular_file(c)) return c.string();
      }
      // Download from the iMoonLab release into the cache under our name.
      fs::path target = models_dir() / our_base;
      const std::string url =
          "https://github.com/iMoonLab/yolov13/releases/download/yolov13/" +
          upstream_base;
      if (run_curl(url, target)) return target.string();
    }
  }

  // 2f) yolo3.pt — convert from upstream yolov3u.pt (anchor-free
  // v3 with v8-style head). No fusion needed — fp16→fp32 cast only.
  if (base == "yolo3.pt" || base == "yolov3u.pt" || base == "yolov3.pt") {
    fs::path target = models_dir() / "yolo3.pt";
    if (fs::exists(target)) return target.string();
    auto try_v3_convert = [&](const fs::path& pt_target) -> bool {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / "yolov3u.pt",
          fs::current_path() / "yolov3u.pt",
          models_dir() / "yolov3u.pt",
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov3_pt(w.string(), pt_target.string());
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
      return false;
    };
    if (try_v3_convert(target)) return target.string();
    fs::path wsrc = models_dir() / "yolov3u.pt";
    const std::string url =
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov3u.pt";
    if (run_curl(url, wsrc) && try_v3_convert(target)) {
      return target.string();
    }
  }

  if (base == "yolo4.pt" || base == "yolov4.pt") {
    fs::path target = models_dir() / "yolo4.pt";
    if (fs::exists(target)) return target.string();
    if (try_v4_convert(target)) return target.string();
    // No local .weights — try downloading from AlexeyAB and converting.
    fs::path wsrc = models_dir() / "yolov4.weights";
    const std::string url =
        "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights";
    if (run_curl(url, wsrc) && try_v4_convert(target)) {
      return target.string();
    }
  }

  // 2g) yolo1.pt — convert from pjreddie's yolov1.weights / yolov1-tiny.weights.
  //     v1 ships with nc=20 (VOC); we default to that. Pure-C++ converter,
  //     no Darknet runtime needed.
  if (base == "yolo1.pt" || base == "yolov1.pt" ||
      base == "yolo1-tiny.pt" || base == "yolov1-tiny.pt") {
    const bool tiny = (base.find("tiny") != std::string::npos);
    fs::path target = models_dir() /
                       (tiny ? "yolo1-tiny.pt" : "yolo1.pt");
    if (fs::exists(target)) return target.string();
    const std::string wname = tiny ? "yolov1-tiny.weights" : "yolov1.weights";
    auto try_v1_convert = [&](const fs::path& pt_target) -> bool {
      std::vector<fs::path> roots = {
          fs::current_path() / "data" / wname,
          fs::current_path() / wname,
          models_dir() / wname,
      };
      for (const auto& w : roots) {
        if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
        std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
        fs::create_directories(pt_target.parent_path());
        try {
          serialization::convert_yolov1_weights(w.string(), pt_target.string(),
                                                 /*nc=*/20);
          return true;
        } catch (const std::exception& e) {
          std::cerr << "[resolve] convert failed: " << e.what() << "\n";
          return false;
        }
      }
      return false;
    };
    if (try_v1_convert(target)) return target.string();
    fs::path wsrc = models_dir() / wname;
    const std::string url = "https://pjreddie.com/media/files/" + wname;
    if (run_curl(url, wsrc) && try_v1_convert(target)) {
      return target.string();
    }
  }

  // 2h) yolo2.pt — convert from pjreddie's yolov2.weights / yolov2-tiny.weights
  //     (COCO, nc=80) or yolov2-voc.weights / yolov2-tiny-voc.weights (VOC,
  //     nc=20). Canonical short form: yolo2{,-tiny}{,-voc}.pt.
  {
    static const std::vector<std::tuple<std::string, std::string,
                                        models::Yolo2Scale, int>> kV2 = {
        // canonical-our,           upstream .weights,            scale,                       nc
        {"yolo2.pt",                "yolov2.weights",             models::Yolo2Scale::Full, 80},
        {"yolov2.pt",               "yolov2.weights",             models::Yolo2Scale::Full, 80},
        {"yolo2-tiny.pt",           "yolov2-tiny.weights",        models::Yolo2Scale::Tiny, 80},
        {"yolov2-tiny.pt",          "yolov2-tiny.weights",        models::Yolo2Scale::Tiny, 80},
        {"yolo2-voc.pt",            "yolov2-voc.weights",         models::Yolo2Scale::Full, 20},
        {"yolov2-voc.pt",           "yolov2-voc.weights",         models::Yolo2Scale::Full, 20},
        {"yolo2-tiny-voc.pt",       "yolov2-tiny-voc.weights",    models::Yolo2Scale::Tiny, 20},
        {"yolov2-tiny-voc.pt",      "yolov2-tiny-voc.weights",    models::Yolo2Scale::Tiny, 20},
    };
    for (const auto& [ours, wname, scale, nc] : kV2) {
      if (base != ours) continue;
      // Canonical cache name strips the optional "v" prefix.
      std::string our_canonical = (ours.rfind("yolov", 0) == 0)
                                       ? ("yolo" + ours.substr(5))
                                       : ours;
      fs::path target = models_dir() / our_canonical;
      if (fs::exists(target)) return target.string();
      auto try_v2_convert = [&](const fs::path& pt_target) -> bool {
        std::vector<fs::path> roots = {
            fs::current_path() / "data" / wname,
            fs::current_path() / wname,
            models_dir() / wname,
        };
        for (const auto& w : roots) {
          if (!fs::exists(w) || !fs::is_regular_file(w)) continue;
          std::cerr << "[resolve] converting " << w << " → " << pt_target << "\n";
          fs::create_directories(pt_target.parent_path());
          try {
            serialization::convert_yolov2_weights(w.string(), pt_target.string(),
                                                   nc, scale);
            return true;
          } catch (const std::exception& e) {
            std::cerr << "[resolve] convert failed: " << e.what() << "\n";
            return false;
          }
        }
        return false;
      };
      if (try_v2_convert(target)) return target.string();
      fs::path wsrc = models_dir() / wname;
      const std::string url = "https://pjreddie.com/media/files/" + wname;
      if (run_curl(url, wsrc) && try_v2_convert(target)) {
        return target.string();
      }
    }
  }
  if (base != spec) {
    // user gave a relative or absolute path; honour it (already missed step 1)
  }
  std::vector<fs::path> candidates = {
      fs::current_path() / "data" / base,
      fs::current_path() / base,
      models_dir() / base,
  };
  for (const auto& c : candidates) {
    if (fs::exists(c) && fs::is_regular_file(c)) return c.string();
  }

  // 3) If recognised upstream name, download.
  //    The local cache is keyed by our canonical `yolo<N>` name, but
  //    upstream still publishes v3..v10 as `yolov<N>...pt`, so we fetch
  //    from the v-prefixed URL and save under the canonical name.
  if (looks_like_upstream_weight(base)) {
    auto target = models_dir() / base;
    std::string upstream = upstream_basename(base);
    // v26 lives in the v8.4.0 asset release; everything else in v8.3.0.
    std::string base_url = kAssetBase;
    if (base.rfind("yolo26", 0) == 0) {
      base_url = "https://github.com/ultralytics/assets/releases/download/v8.4.0";
    }
    if (run_curl(base_url + "/" + upstream, target)) {
      std::cerr << "[resolve] cached " << base << " at " << target
                << " (upstream: " << upstream << ")\n";
      return target.string();
    }
  }

  throw std::runtime_error("could not resolve weights: " + spec +
                           " (checked cwd, ./data, ~/.cache/yolocpp/weights)");
}


// ─── known-dataset registry ──────────────────────────────────────────────
//
// Short-name → URL lookup for `yolocpp download <name>`. URLs follow the
// upstream `v0.0.0` asset-host convention for dataset zips. Adding a new
// dataset is two lines: name + url. To pull from a non-standard host,
// pass the URL directly to `download_known_dataset`.
namespace {
struct DatasetEntry { const char* name; const char* url; };
constexpr DatasetEntry kKnownDatasets[] = {
  {"coco8",     "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"},
  {"coco8-seg", "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip"},
  {"coco8-pose","https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"},
  {"coco128",   "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"},
  {"coco128-seg","https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"},
  {"dota8",     "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8.zip"},
  {"VOC",       "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOC.zip"},
  {"xView",     "https://github.com/ultralytics/assets/releases/download/v0.0.0/xView.zip"},
};

const char* lookup_known_dataset(const std::string& name) {
  for (const auto& e : kKnownDatasets) {
    if (name == e.name) return e.url;
  }
  return nullptr;
}
}  // namespace

std::vector<std::string> known_dataset_names() {
  std::vector<std::string> out;
  for (const auto& e : kKnownDatasets) out.emplace_back(e.name);
  std::sort(out.begin(), out.end());
  return out;
}

std::string download_known_dataset(const std::string& name_or_url) {
  std::string url;
  std::string dest_name;
  if (name_or_url.find("://") != std::string::npos) {
    url = name_or_url;
    auto base = fs::path(url).filename().string();
    auto dot = base.find('.');
    dest_name = (dot == std::string::npos) ? base : base.substr(0, dot);
  } else {
    auto u = lookup_known_dataset(name_or_url);
    if (!u) {
      std::ostringstream msg;
      msg << "unknown dataset '" << name_or_url << "'. Known: ";
      bool first = true;
      for (const auto& n : known_dataset_names()) {
        msg << (first ? "" : ", ") << n;
        first = false;
      }
      msg << ". Or pass a direct URL.";
      throw std::runtime_error(msg.str());
    }
    url = u;
    dest_name = name_or_url;
  }

  fs::path data_dir = fs::current_path() / "data";
  fs::create_directories(data_dir);
  fs::path target_dir = data_dir / dest_name;
  if (fs::exists(target_dir) &&
      (fs::exists(target_dir / "images") ||
       fs::exists(target_dir / "train")  ||
       fs::exists(target_dir / "val")    ||
       !fs::is_empty(target_dir))) {
    std::cerr << "[download] " << dest_name
              << " already present at " << target_dir << "\n";
    return target_dir.string();
  }

  fs::path zip = data_dir / fs::path(url).filename();
  if (!fs::exists(zip) || fs::file_size(zip) == 0) {
    if (!run_curl(url, zip))
      throw std::runtime_error("download failed: " + url);
  }
  if (!run_unzip(zip, data_dir))
    throw std::runtime_error("unzip failed: " + zip.string());

  // Most upstream zips unpack into `<name>/`; fall back to a flat
  // directory with the user's chosen name if the zip's top-level
  // entry has a different shape.
  if (!fs::exists(target_dir)) {
    std::cerr << "[download] note: zip didn't unpack into '" << dest_name
              << "/' — check `" << data_dir << "` for the dataset root\n";
    return data_dir.string();
  }
  return target_dir.string();
}

// Ensure `dst` looks like a YOLO dataset (has `images/` or `train`/`val`).
// If empty/missing and `download_url` is set, fetch + unzip into `dst`'s
// parent (upstream zips conventionally have a single top-level dir
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

// `data=` accepts ONLY a path to a YAML file (upstream `data.yaml` form).
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
  // v7-tiny / v6-tiny etc. — "-tiny" → "tiny".
  if (base.find("-tiny") != std::string::npos) return "tiny";
  // v7-w6 / e6 / d6 / e6e — high-res P6 variants.
  if (base.find("-w6")  != std::string::npos) return "w6";
  if (base.find("-e6e") != std::string::npos) return "e6e";
  if (base.find("-e6")  != std::string::npos) return "e6";
  if (base.find("-d6")  != std::string::npos) return "d6";
  // v6 MBLA variants: yolo6{s,m,l,x}_mbla.pt → "s_mbla" etc.
  for (const std::string& letter : {"s", "m", "l", "x"}) {
    if (base.find(letter + "_mbla") != std::string::npos) {
      return letter + "_mbla";
    }
  }
  // v6 P6 variants: yolo6{n,s,m,l}6.pt or yolov6{n,s,m,l}6.pt → "n6" etc.
  for (const std::string& letter : {"n", "s", "m", "l"}) {
    std::string ours    = "yolo6"  + letter + "6.pt";
    std::string upstream = "yolov6" + letter + "6.pt";
    if (base == ours || base == upstream) return letter + "6";
  }
  // Match canonical "yolo<digits><scale>[u]?(-task)?.pt" — and also accept the
  // legacy upstream "yolov<digits>..." spelling for v3..v10 weights. Scales:
  //   v5/v8/v11/v12/v13/v26: {n, s, m, l, x}
  //   v9 adds {t, c, e}; v10 adds {b}. Including t/c/e here means a
  //   `yolo9t.pt` → scale="t" instead of falling through to the C default
  //   (which silently shape-mismatched 745/782 keys at load time).
  std::smatch m;
  static const std::regex re(
      R"(yolov?[0-9]+([nsmblxtce])u?(?:-(?:cls|seg|pose|obb))?\.pt$)");
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
  if (base.rfind("yolo26", 0) == 0) return "v26";  // upstream official
  if (base.rfind("yolo13", 0) == 0) return "v13";  // Lei et al. (unofficial)
  if (base.rfind("yolo12", 0) == 0) return "v12";  // Tian et al. (unofficial)
  if (base.rfind("yolo11", 0) == 0) return "v11";  // upstream official
  if (base.rfind("yolo10", 0) == 0) return "v10";
  if (base.rfind("yolo9",  0) == 0) return "v9";
  if (base.rfind("yolo8",  0) == 0) return "v8";
  if (base.rfind("yolo7",  0) == 0) return "v7";
  if (base.rfind("yolo6",  0) == 0) return "v6";
  if (base.rfind("yolo5",  0) == 0) return "v5";
  if (base.rfind("yolo4",  0) == 0) return "v4";
  if (base.rfind("yolo3",  0) == 0) return "v3";
  if (base.rfind("yolo2",  0) == 0) return "v2";   // Darknet-era (#67)
  if (base.rfind("yolo1",  0) == 0) return "v1";   // Darknet-era (#66)

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
