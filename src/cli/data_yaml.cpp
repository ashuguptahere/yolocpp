#include "yolocpp/cli/data_yaml.hpp"

#define RYML_SINGLE_HDR_DEFINE_NOW
#include <ryml_all.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace yolocpp::cli {

namespace {

std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open yaml: " + p.string());
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

std::string to_str(ryml::csubstr s) { return std::string(s.str, s.len); }

// Resolve `rel` against `base`. Absolute paths pass through. Returns "" if
// `rel` is empty.
fs::path join(const fs::path& base, const std::string& rel) {
  if (rel.empty()) return {};
  fs::path p(rel);
  if (p.is_absolute()) return fs::weakly_canonical(p);
  return fs::weakly_canonical(base / p);
}

}  // namespace

DataYaml parse_data_yaml(const std::string& yaml_path) {
  fs::path yaml_abs = fs::absolute(yaml_path);
  if (!fs::exists(yaml_abs))
    throw std::runtime_error("data yaml not found: " + yaml_abs.string());

  std::string text = slurp(yaml_abs);
  ryml::Tree tree;
  try {
    tree = ryml::parse_in_arena(ryml::to_csubstr(text));
  } catch (const std::exception& e) {
    throw std::runtime_error("malformed yaml " + yaml_abs.string() + ": "
                             + e.what());
  }
  ryml::ConstNodeRef root = tree.rootref();
  if (!root.is_map())
    throw std::runtime_error("data yaml root is not a mapping: "
                             + yaml_abs.string());

  // Resolve dataset root with fallback chain:
  //   1) absolute `path:`           — use as-is
  //   2) <yaml_dir>/<path>          — Ultralytics-style sibling layout
  //   3) <yaml_dir>                 — yaml lives inside the dataset
  //   4) <yaml_dir>/../<path>       — yaml in datasets/yamls/, data in datasets/<name>/
  // We pick the first candidate that looks like a YOLO dataset (has either
  // `images/` or matches the yaml's own `train:`/`val:` subpaths).
  fs::path yaml_dir = yaml_abs.parent_path();
  std::string path_str;
  if (root.has_child("path") && root["path"].has_val())
    path_str = to_str(root["path"].val());

  std::vector<fs::path> root_candidates;
  if (!path_str.empty()) {
    fs::path pp(path_str);
    if (pp.is_absolute()) {
      root_candidates.push_back(fs::weakly_canonical(pp));
    } else {
      root_candidates.push_back(fs::weakly_canonical(yaml_dir / pp));
      root_candidates.push_back(yaml_dir);
      root_candidates.push_back(fs::weakly_canonical(yaml_dir / ".." / pp));
    }
  } else {
    root_candidates.push_back(yaml_dir);
  }

  fs::path ds_root = root_candidates.front();
  for (const auto& c : root_candidates) {
    if (fs::exists(c / "images") || fs::exists(c / "train") ||
        fs::exists(c / "val")) {
      ds_root = c;
      break;
    }
  }

  DataYaml out;
  out.yaml_path = yaml_abs.string();
  out.root      = ds_root.string();
  if (root.has_child("download") && root["download"].has_val())
    out.download_url = to_str(root["download"].val());

  auto resolve_split = [&](const char* key) -> std::string {
    if (!root.has_child(ryml::to_csubstr(key))) return {};
    auto n = root[ryml::to_csubstr(key)];
    if (n.has_val()) return join(ds_root, to_str(n.val())).string();
    // Sequence form: "[a, b]" → just take the first entry; we don't support
    // multi-source training yet.
    if (n.is_seq() && n.num_children() > 0 && n[0].has_val())
      return join(ds_root, to_str(n[0].val())).string();
    return {};
  };
  out.train_dir = resolve_split("train");
  out.val_dir   = resolve_split("val");
  out.test_dir  = resolve_split("test");

  if (root.has_child("names")) {
    auto n = root["names"];
    if (n.is_map()) {
      // Indexed-map form: { 0: person, 1: bicycle, ... }. Collect into a
      // dense vector covering [0, max_index].
      int max_idx = -1;
      for (auto c : n.children()) {
        if (!c.has_key()) continue;
        try { max_idx = std::max(max_idx, std::stoi(to_str(c.key()))); }
        catch (...) { /* non-integer key — skip */ }
      }
      if (max_idx >= 0) {
        out.names.assign(max_idx + 1, "");
        for (auto c : n.children()) {
          if (!c.has_key() || !c.has_val()) continue;
          int k;
          try { k = std::stoi(to_str(c.key())); }
          catch (...) { continue; }
          if (k >= 0 && k <= max_idx) out.names[k] = to_str(c.val());
        }
      }
    } else if (n.is_seq()) {
      // List form: [person, bicycle, ...].
      for (auto c : n.children()) {
        if (c.has_val()) out.names.push_back(to_str(c.val()));
      }
    }
  }

  return out;
}

}  // namespace yolocpp::cli
