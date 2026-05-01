#include "yolocpp/serialization/yolov10_weights.hpp"

#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

namespace {

struct BNStat {
  at::Tensor weight, bias, running_mean, running_var;
  double     eps = 1e-3;
};
struct ConvBN {
  at::Tensor conv_weight;
  BNStat     bn;
};

std::pair<at::Tensor, at::Tensor> fuse_branch(const ConvBN& cb, int kernel_for_W) {
  auto W = cb.conv_weight.to(torch::kFloat32);
  // Zero-pad 3×3 → 7×7 if needed.
  if (W.size(-1) == 3 && kernel_for_W == 7) {
    W = torch::nn::functional::pad(
        W,
        torch::nn::functional::PadFuncOptions({2, 2, 2, 2}).mode(torch::kConstant).value(0));
  }
  auto gamma  = cb.bn.weight.to(torch::kFloat32);
  auto beta   = cb.bn.bias.to(torch::kFloat32);
  auto mean   = cb.bn.running_mean.to(torch::kFloat32);
  auto var    = cb.bn.running_var.to(torch::kFloat32);
  auto invstd = (var + cb.bn.eps).sqrt().reciprocal();
  auto scale  = gamma * invstd;
  auto W_f    = W * scale.view({-1, 1, 1, 1});
  auto b_f    = beta - mean * scale;
  return {W_f, b_f};
}

// RepVGGDW prefix has both `<prefix>.conv.conv.weight` (the 7×7 dwconv)
// AND `<prefix>.conv1.conv.weight` (the 3×3 dwconv), each with a BN.
// At deploy we collapse the two into a single 7×7 dwconv with bias.
// Returns map prefix → (W [c,1,7,7], b [c]).
std::map<std::string, std::pair<at::Tensor, at::Tensor>>
fuse_all_repvggdw(const std::vector<std::pair<std::string, at::Tensor>>& src) {
  static const std::regex re_branch(
      R"(^(.*)\.(conv|conv1)\.(conv\.weight|bn\.(?:weight|bias|running_mean|running_var))$)");

  struct Pair {
    ConvBN seven, three;
  };
  std::map<std::string, Pair> pairs;

  for (const auto& kv : src) {
    std::smatch m;
    if (!std::regex_match(kv.first, m, re_branch)) continue;
    std::string prefix = m[1].str();
    std::string branch = m[2].str();
    std::string field  = m[3].str();
    auto& p = pairs[prefix];
    ConvBN* dst = (branch == "conv") ? &p.seven : &p.three;
    if      (field == "conv.weight")        dst->conv_weight     = kv.second;
    else if (field == "bn.weight")          dst->bn.weight       = kv.second;
    else if (field == "bn.bias")            dst->bn.bias         = kv.second;
    else if (field == "bn.running_mean")    dst->bn.running_mean = kv.second;
    else if (field == "bn.running_var")     dst->bn.running_var  = kv.second;
  }

  std::map<std::string, std::pair<at::Tensor, at::Tensor>> fused;
  for (auto& [prefix, p] : pairs) {
    if (!p.seven.conv_weight.defined() || !p.three.conv_weight.defined()) continue;
    if (p.seven.conv_weight.size(-1) != 7 || p.three.conv_weight.size(-1) != 3) continue;
    if (p.seven.conv_weight.size(0) != p.seven.conv_weight.size(0)) continue;
    auto [W7, b7] = fuse_branch(p.seven, 7);
    auto [W3, b3] = fuse_branch(p.three, 7);
    fused[prefix] = {W7 + W3, b7 + b3};
  }
  return fused;
}

}  // namespace

int convert_yolov10_pt(const std::string& src_pt_path,
                       const std::string& out_pt_path, int /*nc*/) {
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(), "yolov10: empty state-dict (tried model/ema/'')");

  // 1) RepVGGDW fusion. Note: the regex matches ANY `<prefix>.{conv,conv1}.*`
  // pattern — we filter to only those where conv has 7×7 weight AND conv1
  // has 3×3 weight (i.e. the pair is a real RepVGGDW; not the 3×3+1×1
  // RepConv pattern handled by the v9 converter).
  auto fused = fuse_all_repvggdw(src);
  std::cerr << "[yolov10] fused " << fused.size() << " RepVGGDW pairs\n";

  // Pre-compute fused-prefix set so we know which keys to skip in the
  // pass-through stage.
  std::set<std::string> repvgg_prefixes;
  for (auto& [prefix, _] : fused) repvgg_prefixes.insert(prefix);

  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size());

  // a) Emit fused RepVGGDW blocks as `<prefix>.conv.{weight,bias}`.
  // (Replaces both `<prefix>.conv.conv.weight` + `<prefix>.conv.bn.*` AND
  // `<prefix>.conv1.*`.)
  for (auto& [prefix, wb] : fused) {
    out.emplace_back(prefix + ".conv.weight", wb.first);
    out.emplace_back(prefix + ".conv.bias",   wb.second);
  }

  // b) Pass-through. Skip:
  //    - `num_batches_tracked`
  //    - One2many `model.<i>.cv2.*` and `model.<i>.cv3.*` at the head idx
  //      (we'll rename `one2one_cv2/3` to `cv2/3` instead).
  //    - Inputs of any RepVGGDW pair we just fused (both `<prefix>.conv.*`
  //      and `<prefix>.conv1.*` keys — they're replaced).
  //
  // Detect the head index by looking for the highest layer with
  // `one2one_cv3` keys.
  int head_idx = -1;
  static const std::regex re_one2one(R"(^model\.(\d+)\.one2one_cv3\.)");
  for (auto& kv : src) {
    std::smatch m;
    if (std::regex_search(kv.first, m, re_one2one)) {
      int idx = std::stoi(m[1].str());
      if (idx > head_idx) head_idx = idx;
    }
  }
  std::string head_prefix = "model." + std::to_string(head_idx) + ".";

  for (auto& [name, t] : src) {
    if (name.find("num_batches_tracked") != std::string::npos) continue;
    // Skip one2many head keys (cv2/cv3) — we use one2one only.
    if (head_idx >= 0) {
      if (name.rfind(head_prefix + "cv2.", 0) == 0) continue;
      if (name.rfind(head_prefix + "cv3.", 0) == 0) continue;
    }
    // Skip RepVGGDW input keys (already fused).
    bool skip_repvgg = false;
    for (const auto& p : repvgg_prefixes) {
      if (name.rfind(p + ".conv.",  0) == 0) { skip_repvgg = true; break; }
      if (name.rfind(p + ".conv1.", 0) == 0) { skip_repvgg = true; break; }
    }
    if (skip_repvgg) continue;

    // Rename one2one_cv2/3 → cv2/3 at the head.
    std::string n2 = name;
    if (head_idx >= 0) {
      const std::string from1 = head_prefix + "one2one_cv2.";
      const std::string from2 = head_prefix + "one2one_cv3.";
      if (n2.rfind(from1, 0) == 0) n2 = head_prefix + "cv2." + n2.substr(from1.size());
      else if (n2.rfind(from2, 0) == 0) n2 = head_prefix + "cv3." + n2.substr(from2.size());
    }
    out.emplace_back(n2, t.to(torch::kFloat32));
  }

  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov10] wrote " << out.size() << " tensors to " << out_pt_path
            << "\n";
  return (int)fused.size();
}

int convert_yolov10_dual_pt(const std::string& src_pt_path,
                            const std::string& out_pt_path, int /*nc*/) {
  // Same as convert_yolov10_pt but ALSO routes upstream's
  // `model.<head>.cv2.*` / `cv3.*` (one2many) into our parallel
  // `o2m_detect.cv2.*` / `o2m_detect.cv3.*` slots so that
  // `Yolo10Impl(scale, nc, dual_head=true)` loads with both heads
  // pretrained. RepVGGDW fusion + fp16→fp32 + num_batches_tracked
  // drop are unchanged.
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(),
              "yolov10 dual: empty state-dict (tried model/ema/'')");

  auto fused = fuse_all_repvggdw(src);
  std::cerr << "[yolov10-dual] fused " << fused.size() << " RepVGGDW pairs\n";
  std::set<std::string> repvgg_prefixes;
  for (auto& [prefix, _] : fused) repvgg_prefixes.insert(prefix);

  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size() + 16);
  for (auto& [prefix, wb] : fused) {
    out.emplace_back(prefix + ".conv.weight", wb.first);
    out.emplace_back(prefix + ".conv.bias",   wb.second);
  }

  int head_idx = -1;
  static const std::regex re_one2one(R"(^model\.(\d+)\.one2one_cv3\.)");
  for (auto& kv : src) {
    std::smatch m;
    if (std::regex_search(kv.first, m, re_one2one)) {
      int idx = std::stoi(m[1].str());
      if (idx > head_idx) head_idx = idx;
    }
  }
  std::string head_prefix = "model." + std::to_string(head_idx) + ".";

  for (auto& [name, t] : src) {
    if (name.find("num_batches_tracked") != std::string::npos) continue;
    bool skip_repvgg = false;
    for (const auto& p : repvgg_prefixes) {
      if (name.rfind(p + ".conv.",  0) == 0) { skip_repvgg = true; break; }
      if (name.rfind(p + ".conv1.", 0) == 0) { skip_repvgg = true; break; }
    }
    if (skip_repvgg) continue;

    std::string n2 = name;
    if (head_idx >= 0) {
      const std::string from_o2o_2 = head_prefix + "one2one_cv2.";
      const std::string from_o2o_3 = head_prefix + "one2one_cv3.";
      const std::string from_o2m_2 = head_prefix + "cv2.";
      const std::string from_o2m_3 = head_prefix + "cv3.";
      // one2one → model[head].cv2/cv3 (deploy head, legacy=false)
      if      (n2.rfind(from_o2o_2, 0) == 0) n2 = head_prefix + "cv2." + n2.substr(from_o2o_2.size());
      else if (n2.rfind(from_o2o_3, 0) == 0) n2 = head_prefix + "cv3." + n2.substr(from_o2o_3.size());
      // one2many → o2m_detect.cv2/cv3 (parallel head, legacy=true)
      else if (n2.rfind(from_o2m_2, 0) == 0) n2 = "o2m_detect.cv2." + n2.substr(from_o2m_2.size());
      else if (n2.rfind(from_o2m_3, 0) == 0) n2 = "o2m_detect.cv3." + n2.substr(from_o2m_3.size());
    }
    out.emplace_back(n2, t.to(torch::kFloat32));
  }

  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov10-dual] wrote " << out.size() << " tensors to "
            << out_pt_path << "\n";
  return (int)fused.size();
}

}  // namespace yolocpp::serialization
