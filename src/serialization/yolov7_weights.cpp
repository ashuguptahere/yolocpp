#include "yolocpp/serialization/yolov7_weights.hpp"

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
  // WongKinYiu's RepConv branches use `nn.BatchNorm2d(c)` → PyTorch
  // default eps=1e-5. Other v7 ConvBN modules use upstream-style
  // eps=1e-3 (see the `Conv` module in models/common.py); but those are
  // NOT fused — they're loaded directly with their BN.
  double     eps = 1e-5;
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

// RepConv layers are `model.<i>` where the keys contain `.rbr_dense.` etc.
// Identity branch keys (when present) look like `model.<i>.rbr_identity.<field>`
// where field is `weight`, `bias`, `running_mean`, `running_var`.
//
// rbr_dense / rbr_1x1 are stored as nn.Sequential(Conv, BN); fields:
//   `<prefix>.0.weight`             — Conv weight (no bias)
//   `<prefix>.1.{weight,bias,running_mean,running_var}` — BN
std::map<int, std::pair<at::Tensor, at::Tensor>>
fuse_all_repconv(const std::vector<std::pair<std::string, at::Tensor>>& src) {
  static const std::regex re_branch(
      R"(^model\.(\d+)\.(rbr_dense|rbr_1x1|rbr_identity)\.(.+)$)");

  struct Trio {
    ConvBN dense, one_by_one, identity;
    bool has_identity = false;
  };
  std::map<int, Trio> trios;

  for (const auto& kv : src) {
    std::smatch m;
    if (!std::regex_match(kv.first, m, re_branch)) continue;
    int  layer = std::stoi(m[1].str());
    auto branch = m[2].str();
    auto field  = m[3].str();
    auto& t = trios[layer];
    ConvBN* dst = nullptr;
    if (branch == "rbr_dense")        dst = &t.dense;
    else if (branch == "rbr_1x1")     dst = &t.one_by_one;
    else /* rbr_identity */ {         dst = &t.identity; t.has_identity = true; }

    if (branch == "rbr_identity") {
      if (field == "weight")             dst->bn.weight = kv.second;
      else if (field == "bias")          dst->bn.bias   = kv.second;
      else if (field == "running_mean")  dst->bn.running_mean = kv.second;
      else if (field == "running_var")   dst->bn.running_var  = kv.second;
    } else {
      if (field == "0.weight")            dst->conv_weight = kv.second;
      else if (field == "1.weight")       dst->bn.weight = kv.second;
      else if (field == "1.bias")         dst->bn.bias   = kv.second;
      else if (field == "1.running_mean") dst->bn.running_mean = kv.second;
      else if (field == "1.running_var")  dst->bn.running_var  = kv.second;
    }
  }

  std::map<int, std::pair<at::Tensor, at::Tensor>> fused;
  for (auto& [layer, t] : trios) {
    TORCH_CHECK(t.dense.conv_weight.defined() && t.dense.bn.weight.defined(),
                "RepConv layer ", layer, " missing dense branch");
    int c_out = (int)t.dense.conv_weight.size(0);
    int c_in  = (int)t.dense.conv_weight.size(1);

    auto [W_d, b_d] = fuse_branch(t.dense, c_in, c_out, 3);
    at::Tensor W = W_d, b = b_d;

    if (t.one_by_one.conv_weight.defined()) {
      auto [W_1, b_1] = fuse_branch(t.one_by_one, c_in, c_out, 3);
      W = W + W_1;  b = b + b_1;
    }
    if (t.has_identity) {
      auto [W_i, b_i] = fuse_branch(t.identity, c_in, c_out, 3);
      W = W + W_i;  b = b + b_i;
    }
    fused[layer] = {W, b};
  }
  return fused;
}

}  // namespace

std::vector<std::pair<std::string, at::Tensor>>
reparam_yolov7(const std::vector<std::pair<std::string, at::Tensor>>& src) {
  auto fused = fuse_all_repconv(src);

  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size());

  // a) Emit fused RepConv blocks (model.<i>.conv.{weight,bias})
  for (auto& [layer, wb] : fused) {
    std::string p = "model." + std::to_string(layer);
    out.emplace_back(p + ".conv.weight", wb.first);
    out.emplace_back(p + ".conv.bias",   wb.second);
  }

  // b) Pass-through everything else, casting fp16 → fp32 and skipping
  // RepConv branches + num_batches_tracked. Special case: m1/m2/m3
  // submodules in SPPCSPC (51) — upstream's SPPCSPC names the maxpools
  // simply but our SPPCSPC registers them as m1/m2/m3 too, so direct
  // pass-through works (maxpools have no params, so the only thing
  // shared by name is conv/bn).
  static const std::regex re_skip(
      R"(\.(rbr_dense|rbr_1x1|rbr_identity)\.|num_batches_tracked)");
  for (auto& [name, t] : src) {
    if (std::regex_search(name, re_skip)) continue;
    out.emplace_back(name, t.to(torch::kFloat32));
  }
  return out;
}

int convert_yolov7_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path, int /*nc*/) {
  // Upstream stores under `model` (an attribute of the saved Model
  // wrapper); `ema` is the EMA copy. Try both.
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(), "yolov7: empty state-dict (tried model/ema/'')");
  auto out = reparam_yolov7(src);
  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov7] wrote " << out.size() << " tensors to " << out_pt_path
            << "\n";
  return (int)out.size();
}

}  // namespace yolocpp::serialization
