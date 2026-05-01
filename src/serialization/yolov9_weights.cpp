#include "yolocpp/serialization/yolov9_weights.hpp"

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
  // Ultralytics' Conv (and the BN inside RepConv's conv1/conv2 sub-Convs)
  // uses eps=1e-3.
  double     eps = 1e-3;
};
struct ConvBN {
  at::Tensor conv_weight;
  BNStat     bn;
};

std::pair<at::Tensor, at::Tensor> fuse_branch(const ConvBN& cb, int c_in,
                                              int c_out, int kernel_for_W) {
  at::Tensor W;
  if (cb.conv_weight.defined()) {
    W = cb.conv_weight.to(torch::kFloat32);
  } else {
    TORCH_CHECK(c_in == c_out, "identity branch requires c_in == c_out");
    W = torch::zeros({c_out, c_in, 1, 1}, torch::kFloat32);
    for (int i = 0; i < c_out; ++i) W[i][i][0][0] = 1.0f;
  }
  if (W.size(-1) == 1 && kernel_for_W == 3) {
    W = torch::nn::functional::pad(
        W,
        torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kConstant).value(0));
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

// RepConv keys look like:
//   <prefix>.conv1.conv.weight  (3×3 Conv2d, no bias)
//   <prefix>.conv1.bn.{weight,bias,running_mean,running_var}
//   <prefix>.conv2.conv.weight  (1×1 Conv2d, no bias)
//   <prefix>.conv2.bn.{weight,bias,running_mean,running_var}
//   <prefix>.bn.{...}           (optional identity-BN branch)
//
// `<prefix>` typically ends with ".m.<i>.cv1" inside RepBottleneck —
// that's the only place RepConv appears in v9c.
std::map<std::string, std::pair<at::Tensor, at::Tensor>>
fuse_all_repconv(const std::vector<std::pair<std::string, at::Tensor>>& src) {
  static const std::regex re_branch(
      R"(^(.*)\.(conv1|conv2)\.(conv\.weight|bn\.(?:weight|bias|running_mean|running_var))$)");

  struct Trio {
    ConvBN conv1, conv2;
    BNStat identity_bn;
    bool   has_identity = false;
  };
  std::map<std::string, Trio> trios;

  // Build a quick lookup so we can probe for an optional "<prefix>.bn.*"
  // branch when both conv1 and conv2 sub-blocks are present.
  std::map<std::string, at::Tensor> by_name;
  for (const auto& kv : src) by_name[kv.first] = kv.second;

  for (const auto& kv : src) {
    std::smatch m;
    if (!std::regex_match(kv.first, m, re_branch)) continue;
    std::string prefix = m[1].str();
    std::string branch = m[2].str();
    std::string field  = m[3].str();
    auto& t = trios[prefix];
    ConvBN* dst = (branch == "conv1") ? &t.conv1 : &t.conv2;
    if      (field == "conv.weight")        dst->conv_weight     = kv.second;
    else if (field == "bn.weight")          dst->bn.weight       = kv.second;
    else if (field == "bn.bias")            dst->bn.bias         = kv.second;
    else if (field == "bn.running_mean")    dst->bn.running_mean = kv.second;
    else if (field == "bn.running_var")     dst->bn.running_var  = kv.second;
  }

  // Detect optional identity-BN branch at `<prefix>.bn.*`.
  for (auto& [prefix, t] : trios) {
    auto wn = prefix + ".bn.weight";
    if (by_name.count(wn)) {
      t.has_identity = true;
      t.identity_bn.weight        = by_name[prefix + ".bn.weight"];
      t.identity_bn.bias          = by_name[prefix + ".bn.bias"];
      t.identity_bn.running_mean  = by_name[prefix + ".bn.running_mean"];
      t.identity_bn.running_var   = by_name[prefix + ".bn.running_var"];
    }
  }

  std::map<std::string, std::pair<at::Tensor, at::Tensor>> fused;
  for (auto& [prefix, t] : trios) {
    if (!t.conv1.conv_weight.defined() || !t.conv2.conv_weight.defined()) continue;
    int c_out = (int)t.conv1.conv_weight.size(0);
    int c_in  = (int)t.conv1.conv_weight.size(1);

    auto [W_d, b_d] = fuse_branch(t.conv1, c_in, c_out, 3);
    auto [W_1, b_1] = fuse_branch(t.conv2, c_in, c_out, 3);
    at::Tensor W = W_d + W_1;
    at::Tensor b = b_d + b_1;
    if (t.has_identity) {
      ConvBN id;
      id.bn = t.identity_bn;
      auto [W_i, b_i] = fuse_branch(id, c_in, c_out, 3);
      W = W + W_i;
      b = b + b_i;
    }
    fused[prefix] = {W, b};
  }
  return fused;
}

}  // namespace

int convert_yolov9_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path, int /*nc*/) {
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(), "yolov9: empty state-dict (tried model/ema/'')");

  auto fused = fuse_all_repconv(src);
  std::cerr << "[yolov9] fused " << fused.size() << " RepConv blocks\n";

  // Pre-compute the set of prefixes that have an identity-BN branch so we
  // can skip its keys in the pass-through stage below.
  std::set<std::string> identity_prefixes;
  for (auto& [prefix, _wb] : fused) {
    static const std::regex re_repconv_keys(R"(^.*\.(conv1|conv2)\.)");
    (void)re_repconv_keys;
    identity_prefixes.insert(prefix);
  }

  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size());

  // a) Emit fused RepConv blocks.
  for (auto& [prefix, wb] : fused) {
    out.emplace_back(prefix + ".conv.weight", wb.first);
    out.emplace_back(prefix + ".conv.bias",   wb.second);
  }

  // b) Pass-through everything else.
  static const std::regex re_skip(
      R"(\.(conv1|conv2)\.|num_batches_tracked)");

  for (auto& [name, t] : src) {
    if (std::regex_search(name, re_skip)) continue;
    // Skip identity-BN branch keys (`<prefix>.bn.*`) when `<prefix>` is in
    // identity_prefixes — those have been folded into the fused conv.bias.
    bool skip = false;
    for (const auto& p : identity_prefixes) {
      if (name.rfind(p + ".bn.", 0) == 0) { skip = true; break; }
    }
    if (skip) continue;
    out.emplace_back(name, t.to(torch::kFloat32));
  }

  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov9] wrote " << out.size() << " tensors to " << out_pt_path
            << "\n";
  return (int)fused.size();
}

}  // namespace yolocpp::serialization
