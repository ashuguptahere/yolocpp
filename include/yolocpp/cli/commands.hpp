#pragma once
//
// Internal declarations of the CLI command bodies (the `cmd_*`
// functions implemented in `src/cli/main.cpp`). Both the CLI driver
// itself and the public C++ API (`yolocpp::YOLO` in
// `include/yolocpp/api.hpp`) call into them, so they live in a
// shared namespace rather than `main.cpp`'s anonymous one.
//
// Not part of the user-facing SDK surface — the public entry point
// is `yolocpp::YOLO`. These declarations exist so the api.cpp impl
// can dispatch to the same code paths the CLI uses.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"  // YoloDataset, AugConfig
#include "yolocpp/inference/predictor.hpp"    // Detection

namespace yolocpp::cli {

int cmd_info();

// `out_dets` is optional; when non-null, populated with the
// detections for the LAST processed image (single-input → that
// image; multi-input dir/glob → the last in lex order). #52A2.
int cmd_predict_task(const std::string& task, const std::string& weights,
                     const std::string& source, std::string out, int imgsz,
                     std::string device, std::string scale_s, int nc,
                     float conf, float iou,
                     const std::string& version_hint = "",
                     std::vector<inference::Detection>* out_dets = nullptr);

int cmd_val(const std::string& weights, const std::string& root,
            const std::string& names_csv, int imgsz, std::string device,
            std::string scale_s);

int cmd_val_task(const std::string& task, const std::string& weights,
                 const std::string& data, const std::string& names_csv,
                 int imgsz, const std::string& device,
                 const std::string& scale_s);

int cmd_train(const std::string& root, const std::string& names_csv,
              int imgsz, int epochs, int batch_size, double lr0,
              std::string device, std::string scale_s,
              const std::string& save_dir,
              const std::string& init_weights,
              int patience = 0,
              std::vector<std::pair<std::string, std::string>> args_for_yaml = {},
              std::uint64_t seed = 0);

int cmd_train_task(const std::string& task, const std::string& data,
                   const std::string& names_csv, int imgsz, int epochs,
                   int batch, double lr0, const std::string& device,
                   const std::string& scale_s, const std::string& save_dir,
                   const std::string& weights);

int cmd_export(const std::string& weights, const std::string& format,
               const std::string& out, int imgsz, const std::string& scale_s_in,
               int nc, const std::string& input_name, bool fp16,
               const std::string& version_hint = "",
               const std::string& task = "detect");

int cmd_benchmark(const std::string& weights, const std::string& source,
                  int imgsz, int warmup, int iters,
                  const std::string& cache, const std::string& device);

// Centralised --device validator (also exposed so the API can
// validate device strings the same way the CLI does).
std::string normalise_device(std::string d);

// Split a comma-separated string into trimmed tokens. Used by the
// CLI dispatcher (`--export-after-train onnx,trt` etc.) and by the
// API; defined alongside the cmd_* functions in commands.cpp.
std::vector<std::string> split_csv(const std::string& s);

// Format-aware dataset factory (#54B → CLI). Auto-detects:
//   - `<spec>.csv` / `.tsv`                     → FlatDataset
//   - `<spec>.json`                             → CocoDataset (images_dir
//                                                  defaults to <spec>'s parent)
//   - `<spec>/JPEGImages` + `Annotations`       → VocDataset
//   - `<spec>/images/<split>` + `labels/...`    → YoloDataset (existing)
//   - `<spec>.yaml` / `.yml`                    → resolve via data.yaml
//                                                  (root is the resolved dir)
// All four route through `YoloDataset`'s pre-loaded ctor so trainer
// + validator stay typed on a single concrete dataset class. `names`
// supplies class names for formats that don't carry their own
// (YOLO + Flat); ignored for COCO + VOC where names come from the
// dataset itself (COCO categories / VOC default-20).
yolocpp::datasets::YoloDataset make_dataset(
    const std::string& spec, const std::string& split, int imgsz,
    const std::vector<std::string>& names,
    const yolocpp::datasets::AugConfig& aug = {},
    std::uint64_t seed = 0);

}  // namespace yolocpp::cli
