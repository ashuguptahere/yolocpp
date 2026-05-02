#pragma once
//
// Reader for upstream-style `.pt` checkpoints (Python torch.save output).
//
// torch.save produces a zip archive containing:
//   <prefix>/data.pkl       — the top-level Python pickle
//   <prefix>/data/<id>      — raw tensor storage bytes
// where <prefix> is the zip's first path component (often the model name).
//
// Loading proceeds in two stages:
//   1. unpickle data.pkl using libtorch's Unpickler with a permissive
//      TypeResolver/ObjLoader so that Python-only classes (DetectionModel,
//      ParameterDict, etc.) become opaque generic Objects with their
//      __setstate__ payload preserved.
//   2. recursively walk the resulting IValue tree and flatten any
//      nn.Module-shaped subtree into a state_dict — a flat dict of
//      "<module>.<sub>...<param>" → at::Tensor.
//

#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

namespace yolocpp::serialization {

// Flattened (name, tensor) pairs in iteration order matching PyTorch's
// nn.Module.state_dict() — i.e. own _parameters first, then _buffers,
// then recurse into _modules in registered order.
struct StateDict {
  std::vector<std::pair<std::string, at::Tensor>> entries;

  bool          contains(const std::string& key) const;
  at::Tensor    at(const std::string& key) const;
  std::size_t   size() const { return entries.size(); }
};

// Loads an upstream-style `.pt` and returns the model's state_dict.
//
//   pt_path:  absolute or relative path to the .pt file.
//   submodel: dotted path inside the checkpoint to the actual nn.Module
//             (default "model" — upstream stores the model under that key).
//
// Throws std::runtime_error on any error.
StateDict load_state_dict(const std::string& pt_path,
                          const std::string& submodel = "model");

}  // namespace yolocpp::serialization
