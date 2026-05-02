#pragma once
//
// RF-DETR 1.6.5 weight converter (#65E2 first slice).
//
// The upstream checkpoint format is `torch.save({'model': flat_state_dict,
// 'optimizer': ..., 'epoch': ...}, path)` — already-flattened dotted-key
// dict-of-tensors. Loading proceeds in three steps:
//
//   1. `serialization::load_flat_state_dict(path, "model")` returns the
//      flat dict (12 official files validated under `tests/test_rfdetr_pt_load.cpp`).
//   2. This function iterates the dict and, for each upstream key,
//      looks up a same-named parameter on the destination module via
//      `named_parameters()`. If found and the shape matches, copy
//      bytes in-place. If the key isn't registered yet (because the
//      corresponding module rewrite under #65A2..D2 hasn't landed),
//      it's reported as "unmatched" — NOT an error during the
//      transition.
//   3. Returns a coverage report so the caller can assert progress.
//
// Once #65A2..D2 land, every key matches and we promote the unmatched
// list to an error.

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolocpp::serialization {

struct RFDetrLoadReport {
  int matched   = 0;     // keys copied 1-to-1 onto destination params
  int shape_mismatch = 0;
  int unmatched = 0;     // keys present in upstream but not in our model
  int missing   = 0;     // params in our model not provided by upstream
  std::vector<std::string> unmatched_keys;       // first 8 for diagnostics
  std::vector<std::string> shape_mismatch_keys;  // first 8 for diagnostics
  std::vector<std::string> missing_keys;         // first 8 for diagnostics

  std::string summary() const;
};

// Load the upstream RF-DETR `.pt`/`.pth` file at `pt_path` and copy
// each upstream tensor into the matching parameter on `module` via
// `named_parameters()`. Returns a coverage report.
//
// Pre #65A2..D2 land, this is non-throwing: it simply reports how many
// upstream keys had no destination parameter. Post-#65D2 a
// `strict=true` mode (default once the transition completes) will
// throw on any unmatched / mismatched key.
RFDetrLoadReport load_rfdetr_pt(const std::string& pt_path,
                                  torch::nn::Module& module,
                                  bool strict = false);

}  // namespace yolocpp::serialization
