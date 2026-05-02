// RF-DETR weight converter implementation. See header for the
// transition plan; this file is intentionally module-agnostic — it
// pulls the flat state_dict and binds onto whatever
// `named_parameters()` returns. The architecture rewrite (#65A2..D2)
// changes which keys actually find a home; this loader doesn't
// care.

#include "yolocpp/serialization/rfdetr_weights.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "yolocpp/serialization/pt_loader.hpp"

namespace yolocpp::serialization {

std::string RFDetrLoadReport::summary() const {
  std::ostringstream os;
  os << "rfdetr-load: matched=" << matched
     << " unmatched=" << unmatched
     << " shape_mismatch=" << shape_mismatch
     << " missing=" << missing;
  if (!unmatched_keys.empty()) {
    os << "\n  first unmatched upstream keys:";
    for (const auto& k : unmatched_keys) os << "\n    " << k;
  }
  if (!shape_mismatch_keys.empty()) {
    os << "\n  shape-mismatched keys:";
    for (const auto& k : shape_mismatch_keys) os << "\n    " << k;
  }
  if (!missing_keys.empty()) {
    os << "\n  first model params with no upstream value:";
    for (const auto& k : missing_keys) os << "\n    " << k;
  }
  return os.str();
}

RFDetrLoadReport load_rfdetr_pt(const std::string& pt_path,
                                  torch::nn::Module& module,
                                  bool strict) {
  auto sd = load_flat_state_dict(pt_path, /*submodel=*/"model");

  // Build a lookup of upstream key → tensor for O(1) access.
  std::unordered_map<std::string, at::Tensor> src;
  src.reserve(sd.entries.size());
  for (auto& [k, t] : sd.entries) src.emplace(k, t);

  // Iterate destination parameters; copy by name.
  RFDetrLoadReport rep;
  std::vector<std::string> seen;
  seen.reserve(sd.entries.size());

  torch::NoGradGuard ng;
  auto params = module.named_parameters(/*recurse=*/true);
  std::unordered_map<std::string, at::Tensor*> dst;
  dst.reserve(params.size());
  for (auto& kv : params) {
    dst.emplace(kv.key(), &kv.value());
  }
  // Buffers (non-trainable persistent state, e.g. position_embeddings
  // saved as buffer in some HF wrappers).
  auto buffers = module.named_buffers(/*recurse=*/true);
  for (auto& kv : buffers) dst.emplace(kv.key(), &kv.value());

  for (auto& [k, t] : sd.entries) {
    auto it = dst.find(k);
    if (it == dst.end()) {
      rep.unmatched++;
      if (rep.unmatched_keys.size() < 8) rep.unmatched_keys.push_back(k);
      continue;
    }
    if (it->second->sizes() != t.sizes()) {
      rep.shape_mismatch++;
      if (rep.shape_mismatch_keys.size() < 8) {
        std::ostringstream os;
        os << k << " upstream=" << t.sizes() << " ours=" << it->second->sizes();
        rep.shape_mismatch_keys.push_back(os.str());
      }
      continue;
    }
    it->second->copy_(t.to(it->second->dtype()));
    rep.matched++;
    seen.push_back(k);
  }

  // Find destination params with no upstream match.
  std::sort(seen.begin(), seen.end());
  for (auto& [name, ptr] : dst) {
    if (!std::binary_search(seen.begin(), seen.end(), name)) {
      rep.missing++;
      if (rep.missing_keys.size() < 8) rep.missing_keys.push_back(name);
    }
  }

  if (strict && (rep.unmatched > 0 || rep.shape_mismatch > 0 ||
                  rep.missing > 0)) {
    throw std::runtime_error(
        "load_rfdetr_pt(strict=true): " + rep.summary());
  }
  return rep;
}

}  // namespace yolocpp::serialization
