#include "yolocpp/serialization/yolov6_weights.hpp"

#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

namespace {

// Apply RepVGG re-parameterization on a single block:
//   dense      : 3×3 Conv + BN
//   one_by_one : 1×1 Conv + BN
//   identity   : optional pure BN (used only when in_c == out_c && s == 1)
//
// Returns (W: [c_out, c_in, 3, 3] float32, b: [c_out] float32).
//
// Math (per branch i): fused = γ_i / sqrt(σ_i² + ε); then
//   W_fused_i = W_i * fused (broadcast on out-channel axis)
//   b_fused_i = β_i - μ_i * fused
// then sum the three branches in W and b.
//
// 1×1 branch's W is zero-padded to 3×3 (so it sums correctly into the 3×3
// kernel). Identity branch's W is constructed as a per-output-channel 1×1
// identity matrix at out_c==in_c, then BN-fused, then zero-padded to 3×3.
struct BNStat {
  at::Tensor weight, bias, running_mean, running_var;
  // Meituan's RepVGGBlock branches use plain `nn.BatchNorm2d(out_c)` with
  // PyTorch's default `eps=1e-5`. (ConvBNReLU elsewhere uses 1e-3, but
  // those modules are NOT fused — they're loaded directly.) Getting this
  // wrong cascades through the entire RepVGG-heavy network and produces
  // near-uniform cls scores.
  double     eps = 1e-5;
};
struct ConvBN {
  at::Tensor conv_weight;   // optional (identity has none)
  BNStat     bn;
};
std::pair<at::Tensor, at::Tensor> fuse_branch(const ConvBN& cb, int c_in,
                                              int c_out, int kernel_for_W) {
  at::Tensor W;
  if (cb.conv_weight.defined()) {
    W = cb.conv_weight.to(torch::kFloat32);
  } else {
    // Identity branch: build [c_out, c_in, 1, 1] identity tensor.
    TORCH_CHECK(c_in == c_out, "identity branch requires c_in == c_out");
    W = torch::zeros({c_out, c_in, 1, 1}, torch::kFloat32);
    for (int i = 0; i < c_out; ++i) W[i][i][0][0] = 1.0f;
  }
  // Pad to 3×3 if needed.
  if (W.size(-1) == 1 && kernel_for_W == 3) {
    W = torch::nn::functional::pad(
        W,
        torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kConstant).value(0));
  }
  auto gamma  = cb.bn.weight.to(torch::kFloat32);
  auto beta   = cb.bn.bias.to(torch::kFloat32);
  auto mean   = cb.bn.running_mean.to(torch::kFloat32);
  auto var    = cb.bn.running_var.to(torch::kFloat32);
  auto invstd = (var + cb.bn.eps).sqrt().reciprocal();    // [c_out]
  auto scale  = gamma * invstd;                            // [c_out]
  auto W_f    = W * scale.view({-1, 1, 1, 1});             // broadcast
  auto b_f    = beta - mean * scale;                       // [c_out]
  return {W_f, b_f};
}

// Walks the upstream state-dict, collects every RepVGG branch trio,
// and returns a map <repvgg_prefix> → fused (W, b).
//
// `repvgg_prefix` is the part of the key preceding `.rbr_dense.` —
// e.g. "backbone.stem", "backbone.ERBlock_2.0",
// "backbone.ERBlock_2.1.conv1", "backbone.ERBlock_2.1.block.0", etc.
//
// `stride==2` is detected by inspecting the dense branch weight's spatial
// dimensions and the parent module name (downsamples sit at `.0` of an
// `ERBlock_N`). For RepVGG re-parameterization the stride doesn't change
// the formula — we just trust the dense branch's stride.
std::map<std::string, std::pair<at::Tensor, at::Tensor>>
fuse_all_repvgg(const std::vector<std::pair<std::string, at::Tensor>>& src) {
  // Group entries by repvgg_prefix and branch.
  struct Trio {
    ConvBN dense;
    ConvBN one_by_one;
    ConvBN identity;
    bool has_identity = false;
  };
  std::map<std::string, Trio> trios;

  static const std::regex re_branch(
      R"(^(.*)\.(rbr_dense|rbr_1x1|rbr_identity)\.(.+)$)");

  for (const auto& kv : src) {
    std::smatch m;
    if (!std::regex_match(kv.first, m, re_branch)) continue;
    const std::string prefix = m[1].str();
    const std::string branch = m[2].str();
    const std::string field  = m[3].str();
    auto& t = trios[prefix];
    ConvBN* target = nullptr;
    if (branch == "rbr_dense")        target = &t.dense;
    else if (branch == "rbr_1x1")     target = &t.one_by_one;
    else /* rbr_identity */ {         target = &t.identity; t.has_identity = true; }

    // Field paths look like:
    //   conv.weight        — 3×3 / 1×1 dense conv kernel
    //   bn.weight, bn.bias, bn.running_mean, bn.running_var
    // For rbr_identity (BN-only):
    //   weight, bias, running_mean, running_var
    if (field == "conv.weight")            target->conv_weight  = kv.second;
    else if (field == "bn.weight")          target->bn.weight    = kv.second;
    else if (field == "bn.bias")            target->bn.bias      = kv.second;
    else if (field == "bn.running_mean")    target->bn.running_mean = kv.second;
    else if (field == "bn.running_var")     target->bn.running_var  = kv.second;
    else if (field == "weight" && branch == "rbr_identity")
      target->bn.weight = kv.second;
    else if (field == "bias" && branch == "rbr_identity")
      target->bn.bias = kv.second;
    else if (field == "running_mean" && branch == "rbr_identity")
      target->bn.running_mean = kv.second;
    else if (field == "running_var" && branch == "rbr_identity")
      target->bn.running_var = kv.second;
    // num_batches_tracked etc. ignored.
  }

  std::map<std::string, std::pair<at::Tensor, at::Tensor>> fused;
  for (auto& [prefix, t] : trios) {
    TORCH_CHECK(t.dense.conv_weight.defined() &&
                t.dense.bn.weight.defined(),
                "RepVGG block at ", prefix, " missing dense branch");
    int c_out = (int)t.dense.conv_weight.size(0);
    int c_in  = (int)t.dense.conv_weight.size(1);

    auto [W_d, b_d] = fuse_branch(t.dense, c_in, c_out, 3);
    at::Tensor W = W_d, b = b_d;

    if (t.one_by_one.conv_weight.defined()) {
      auto [W_1, b_1] = fuse_branch(t.one_by_one, c_in, c_out, 3);
      W = W + W_1;
      b = b + b_1;
    }
    if (t.has_identity) {
      auto [W_i, b_i] = fuse_branch(t.identity, c_in, c_out, 3);
      W = W + W_i;
      b = b + b_i;
    }
    fused[prefix] = {W, b};
  }
  return fused;
}

// Translate upstream prefixes to ours (e.g. "ERBlock_2.0" → "ERBlock_2_down").
std::string rename_prefix(std::string prefix) {
  // backbone.ERBlock_N.0 → backbone.ERBlock_N_down
  // backbone.ERBlock_N.1 → backbone.ERBlock_N_block
  // backbone.ERBlock_5.2.cspsppf → backbone.ERBlock_5_cspsppf
  static const std::vector<std::pair<std::regex, std::string>> rules = {
      // P6 (n6/s6/m6/l6): SPP lives at ERBlock_6.2 instead of ERBlock_5.2.
      {std::regex(R"(^backbone\.ERBlock_6\.2\.cspsppf)"), "backbone.ERBlock_6_cspsppf"},
      {std::regex(R"(^backbone\.ERBlock_6\.2\.sppf)"),    "backbone.ERBlock_6_cspsppf.sppf"},
      // v6n/s have CSPSPPF at .2.cspsppf — register as ERBlock_5_cspsppf.
      {std::regex(R"(^backbone\.ERBlock_5\.2\.cspsppf)"), "backbone.ERBlock_5_cspsppf"},
      // v6m/l have SimSPPF at .2.sppf — preserve the inner `sppf` infix
      // (our SimSPPF wrapper has child "sppf").
      {std::regex(R"(^backbone\.ERBlock_5\.2\.sppf)"),    "backbone.ERBlock_5_cspsppf.sppf"},
      {std::regex(R"(^backbone\.ERBlock_(\d)\.0)"),       "backbone.ERBlock_$1_down"},
      {std::regex(R"(^backbone\.ERBlock_(\d)\.1)"),       "backbone.ERBlock_$1_block"},
  };
  for (const auto& [re, repl] : rules) {
    auto out = std::regex_replace(prefix, re, repl);
    if (out != prefix) return out;
  }
  return prefix;
}

}  // namespace

int convert_yolov6_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path, int /*nc*/) {
  // Upstream stores everything under `model.<...>` — but Meituan's writer
  // puts it under different top-level keys depending on save path. Probe
  // common roots; bare-form (no submodel prefix) is also accepted.
  std::vector<std::pair<std::string, at::Tensor>> src;
  for (const std::string& root : {"model", "ema", ""}) {
    try {
      auto sd = load_state_dict(src_pt_path, root);
      if (!sd.entries.empty()) { src = std::move(sd.entries); break; }
    } catch (...) {}
  }
  TORCH_CHECK(!src.empty(), "yolov6: empty state-dict (tried model/ema/'')");

  // 1) RepVGG fusion across all blocks.
  auto fused = fuse_all_repvgg(src);
  std::cerr << "[yolov6] fused " << fused.size() << " RepVGG blocks\n";

  // 2) Walk source entries and emit converted ones.
  std::vector<std::pair<std::string, at::Tensor>> out;
  out.reserve(src.size());

  // a) Emit fused RepVGG blocks: each becomes <newprefix>.conv.{weight,bias}.
  std::set<std::string> rep_emitted;
  for (auto& [pref, wb] : fused) {
    std::string new_prefix = rename_prefix(pref);
    out.emplace_back(new_prefix + ".conv.weight", wb.first);
    out.emplace_back(new_prefix + ".conv.bias",   wb.second);
    rep_emitted.insert(pref);
  }

  // b) Emit non-RepVGG entries with the ConvBNReLU `.block.` wrapper
  // stripped + prefix renamed. Only strip `.block.` when followed by
  // `conv.` or `bn.` — otherwise we'd also remove RepBlock's `.block.<i>.`
  // ModuleList path (used in v6l's BepC3.m), which we want to keep.
  static const std::regex re_block(R"(\.block\.(?=(conv|bn)\.))");
  static const std::regex re_skip_repvgg(
      R"(\.(rbr_dense|rbr_1x1|rbr_identity)\.)");
  static const std::regex re_skip_misc(R"(num_batches_tracked|^proj_conv|^detect\.proj$)");

  for (auto& [name, t] : src) {
    if (std::regex_search(name, re_skip_repvgg)) continue;
    if (std::regex_search(name, re_skip_misc))    continue;

    std::string s = std::regex_replace(name, re_block, ".");
    s = rename_prefix(s);
    out.emplace_back(s, t.to(torch::kFloat32));
  }

  // detect.proj and detect.proj_conv.weight are deterministic (arange / its
  // reshape) — our model recreates them at construction. Skipping is fine.

  save_state_dict(out_pt_path, out);
  std::cerr << "[yolov6] wrote " << out.size() << " tensors to " << out_pt_path
            << "\n";
  return (int)fused.size();
}

}  // namespace yolocpp::serialization
