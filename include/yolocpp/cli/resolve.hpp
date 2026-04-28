#pragma once
//
// Auto-resolution for CLI arguments that name weights / datasets.
//
// Behaviour matches user expectations from Ultralytics:
//   model=yolo8n.pt   → looks in cwd, ./data, ~/.cache/yolocpp/weights;
//                         downloads from ultralytics/assets if recognised
//   data=coco8         → looks in cwd, ./data; downloads coco8 zip
//   data=coco          → ditto; downloads val2017 images + YOLO labels
//

#include <string>

namespace yolocpp::cli {

// Returns the local path to the weights file. May download to
// ./data/<basename> or ~/.cache/yolocpp/weights/<basename> if absent.
// Throws std::runtime_error if it can't find or fetch.
std::string resolve_weights(const std::string& spec);

// Returns the local dataset root directory. May download to ./data/<name>.
std::string resolve_dataset(const std::string& spec);

// Infer YOLO8 scale ("n"/"s"/"m"/"l"/"x") from a weights filename like
// "yolo8s-seg.pt". Returns empty string if not recognised.
std::string scale_from_filename(const std::string& path);

// Infer YOLO version family from a weights filename. Recognised values:
//   "v3", "v5", "v6", "v7", "v8", "v9", "v10",
//   "v11" (Ultralytics official, ships as `yolo11*.pt` — no 'v'),
//   "v12" (Tian et al., unofficial — `yolo12*.pt`),
//   "v13" (Lei et al., unofficial — `yolo13*.pt`),
//   "v26" (Ultralytics official, ships as `yolo26*.pt` — no 'v'),
//   "rtdetr".
// Returns "v8" as a permissive default — most Ultralytics-style anchor-free
// weights load through the v8 path since their Detect head shares the same
// DFL layout.
std::string version_from_filename(const std::string& path);

}  // namespace yolocpp::cli
