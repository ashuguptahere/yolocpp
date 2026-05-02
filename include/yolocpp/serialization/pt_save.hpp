#pragma once
//
// Write a state_dict to a .pt file in a layout our pt_loader can read back.
//
// The loader expects an upstream-shape archive:
//   data.pkl unpickles to GenericDict { "model": Object{ ... state ... } }
// and the model state has _modules/_parameters/_buffers nested OrderedDicts
// pointing at storage records data/<id>.
//
// We emit a *minimal* such archive by hand:
//   • flat _parameters dict at the top of the model, no _modules
//   • each tensor is a _rebuild_tensor_v2 call referencing storage data/N
//   • zip records: data.pkl + data/0..data/N
//
// This keeps the round-trip purely C++ and lets `Predictor` and `mode=val`
// load trainer outputs the same way they load upstream weights.
//

#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

namespace yolocpp::serialization {

// Save state-dict entries (name → tensor) as a .pt file readable by
// load_state_dict(path).
void save_state_dict(
    const std::string& path,
    const std::vector<std::pair<std::string, at::Tensor>>& entries);

}  // namespace yolocpp::serialization
