#pragma once
//
// Auto-resolution for CLI arguments that name weights / datasets.
//
// Behaviour matches user expectations from upstream tooling:
//   model=yolo8n.pt   → looks in cwd, ./data, ~/.cache/yolocpp/weights;
//                         downloads from the upstream asset host if recognised
//   data=coco8         → looks in cwd, ./data; downloads coco8 zip
//   data=coco          → ditto; downloads val2017 images + YOLO labels
//

#include <string>
#include <vector>

namespace yolocpp::cli {

// Returns the local path to the weights file. May download to
// ./data/<basename> or ~/.cache/yolocpp/weights/<basename> if absent.
// Throws std::runtime_error if it can't find or fetch.
std::string resolve_weights(const std::string& spec);

// Returns the local dataset root directory. May download to ./data/<name>.
std::string resolve_dataset(const std::string& spec);

// Download a well-known dataset by short name (e.g. "coco8", "coco128",
// "voc", "dota8") into `./data/<name>/` and return the resulting
// directory path. If `name_or_url` looks like a URL (contains "://"),
// the URL is fetched directly into `./data/<filename-stem>/`.
//
// Throws std::runtime_error on unknown name, network failure, or
// unzip failure.
std::string download_known_dataset(const std::string& name_or_url);

// List the short names of every known dataset (alphabetical).
std::vector<std::string> known_dataset_names();

// Infer YOLO8 scale ("n"/"s"/"m"/"l"/"x") from a weights filename like
// "yolo8s-seg.pt". Returns empty string if not recognised.
std::string scale_from_filename(const std::string& path);

// Infer YOLO version family from a weights filename. Recognised values:
//   "v3", "v5", "v6", "v7", "v8", "v9", "v10",
//   "v11" (upstream official, ships as `yolo11*.pt` — no 'v'),
//   "v12" (Tian et al., unofficial — `yolo12*.pt`),
//   "v13" (Lei et al., unofficial — `yolo13*.pt`),
//   "v26" (upstream official, ships as `yolo26*.pt` — no 'v'),
//   "rtdetr".
// Returns "v8" as a permissive default — most anchor-free weights in
// the wild load through the v8 path since their Detect head shares the
// same DFL layout.
std::string version_from_filename(const std::string& path);

}  // namespace yolocpp::cli
