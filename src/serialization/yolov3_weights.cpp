#include "yolocpp/serialization/yolov3_weights.hpp"

#include <iostream>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

int convert_yolov3_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path, int /*nc*/) {
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(), "yolov3: empty state-dict (tried model/ema/'')");

  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size());
  for (auto& [name, t] : src) {
    if (name.find("num_batches_tracked") != std::string::npos) continue;
    out.emplace_back(name, t.to(torch::kFloat32));
  }

  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov3] wrote " << out.size() << " tensors to " << out_pt_path
            << "\n";
  return (int)out.size();
}

}  // namespace yolocpp::serialization
