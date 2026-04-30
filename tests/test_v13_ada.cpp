// Parity test: AdaHyperedgeGen forward output vs the Python-dumped
// activation from yolo13n's HyperACE branch1.edge_generator.
//
// Reads:
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_inp.bin                (1 × 1600 × 64)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_out.bin                (1 × 1600 × 4)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_prototype_base.bin     (4 × 64)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_context_net_weight.bin (256 × 128)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_context_net_bias.bin   (256)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_pre_head_proj_weight.bin(64 × 64)
//   /tmp/yolocpp_parity/dumps/yolo13n/ada_b1_pre_head_proj_bias.bin (64)
//
// Emits exit code 0 on parity (max abs diff < 1e-5), 1 otherwise.
// If the dumps are absent (Python harness not run), test is skipped (exit 0).

#include "yolocpp/models/yolo13.hpp"

#include <torch/torch.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static torch::Tensor read_bin(const std::string& path,
                              std::vector<int64_t> shape) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int64_t numel = 1;
  for (auto d : shape) numel *= d;
  std::vector<float> buf(numel);
  f.read(reinterpret_cast<char*>(buf.data()),
         numel * (int64_t)sizeof(float));
  if (!f) throw std::runtime_error("short read on " + path);
  return torch::from_blob(buf.data(), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

static int run_one(const std::string& base, const std::string& tag) {
  if (!fs::exists(base + "/ada_" + tag + "_inp.bin")) {
    std::printf("[skip] %s/ada_%s_inp.bin missing\n", base.c_str(),
                tag.c_str());
    return 0;
  }
  auto inp_ref = read_bin(base + "/ada_" + tag + "_inp.bin", {1, 1600, 64});
  auto out_ref = read_bin(base + "/ada_" + tag + "_out.bin", {1, 1600, 4});
  auto proto   = read_bin(base + "/ada_" + tag + "_prototype_base.bin", {4, 64});
  auto ctx_w   = read_bin(base + "/ada_" + tag + "_context_net_weight.bin",
                          {256, 128});
  auto ctx_b   = read_bin(base + "/ada_" + tag + "_context_net_bias.bin",
                          {256});
  auto php_w   = read_bin(base + "/ada_" + tag + "_pre_head_proj_weight.bin",
                          {64, 64});
  auto php_b   = read_bin(base + "/ada_" + tag + "_pre_head_proj_bias.bin",
                          {64});

  yolocpp::models::AdaHyperedgeGen ada(/*node_dim=*/64, /*num_hyperedges=*/4,
                                        /*num_heads=*/4, /*context=*/"both");
  torch::NoGradGuard ng;
  ada->prototype_base.copy_(proto);
  ada->context_net->weight.copy_(ctx_w);
  ada->context_net->bias.copy_(ctx_b);
  ada->pre_head_proj->weight.copy_(php_w);
  ada->pre_head_proj->bias.copy_(php_b);
  ada->eval();

  auto out_ours = ada->forward(inp_ref);
  if (!out_ours.sizes().equals(out_ref.sizes())) {
    std::printf("[fail] %s: shape mismatch\n", tag.c_str());
    return 1;
  }
  auto diff = (out_ours - out_ref).abs();
  double max_abs = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  std::printf("[v13/ada] %s: max|Δ|=%.3e mean|Δ|=%.3e\n", tag.c_str(),
              max_abs, mean_abs);
  if (max_abs > 1e-5) {
    std::printf("[fail] %s: max|Δ| > 1e-5\n", tag.c_str());
    return 1;
  }
  return 0;
}

// AdaHGConv parity: load weights for the full hgnn module (edge_generator
// + edge_proj.0 + node_proj.0), forward the dumped input, compare.
static int run_hgconv(const std::string& base, const std::string& tag) {
  if (!fs::exists(base + "/" + tag + "_inp.bin")) {
    std::printf("[skip] %s/%s_inp.bin missing\n", base.c_str(), tag.c_str());
    return 0;
  }
  auto inp_ref = read_bin(base + "/" + tag + "_inp.bin", {1, 1600, 64});
  auto out_ref = read_bin(base + "/" + tag + "_out.bin", {1, 1600, 64});

  auto proto   = read_bin(base + "/" + tag +
                          "_edge_generator_prototype_base.bin", {4, 64});
  auto ctx_w   = read_bin(base + "/" + tag +
                          "_edge_generator_context_net_weight.bin",
                          {256, 128});
  auto ctx_b   = read_bin(base + "/" + tag +
                          "_edge_generator_context_net_bias.bin", {256});
  auto php_w   = read_bin(base + "/" + tag +
                          "_edge_generator_pre_head_proj_weight.bin",
                          {64, 64});
  auto php_b   = read_bin(base + "/" + tag +
                          "_edge_generator_pre_head_proj_bias.bin", {64});
  auto ep_w    = read_bin(base + "/" + tag + "_edge_proj_0_weight.bin",
                          {64, 64});
  auto ep_b    = read_bin(base + "/" + tag + "_edge_proj_0_bias.bin", {64});
  auto np_w    = read_bin(base + "/" + tag + "_node_proj_0_weight.bin",
                          {64, 64});
  auto np_b    = read_bin(base + "/" + tag + "_node_proj_0_bias.bin", {64});

  yolocpp::models::AdaHGConv hg(/*embed_dim=*/64, /*num_hyperedges=*/4,
                                 /*num_heads=*/4, /*context=*/"both");
  torch::NoGradGuard ng;
  hg->edge_generator->prototype_base.copy_(proto);
  hg->edge_generator->context_net->weight.copy_(ctx_w);
  hg->edge_generator->context_net->bias.copy_(ctx_b);
  hg->edge_generator->pre_head_proj->weight.copy_(php_w);
  hg->edge_generator->pre_head_proj->bias.copy_(php_b);
  // Reach into the Sequential by index (Linear is child "0").
  auto* ep_lin = hg->edge_proj[0]->as<torch::nn::LinearImpl>();
  auto* np_lin = hg->node_proj[0]->as<torch::nn::LinearImpl>();
  ep_lin->weight.copy_(ep_w);
  ep_lin->bias.copy_(ep_b);
  np_lin->weight.copy_(np_w);
  np_lin->bias.copy_(np_b);
  hg->eval();

  auto out_ours = hg->forward(inp_ref);
  if (!out_ours.sizes().equals(out_ref.sizes())) {
    std::printf("[fail] %s shape mismatch\n", tag.c_str());
    return 1;
  }
  auto diff = (out_ours - out_ref).abs();
  double max_abs  = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  std::printf("[v13/hgconv] %s: max|Δ|=%.3e mean|Δ|=%.3e\n",
              tag.c_str(), max_abs, mean_abs);
  if (max_abs > 1e-5) {
    std::printf("[fail] %s: max|Δ| > 1e-5\n", tag.c_str());
    auto a = out_ours.flatten();
    auto b = out_ref.flatten();
    for (int i = 0; i < 6; ++i) {
      std::printf("  [%d] ours=%.7f ref=%.7f\n", i,
                  a[i].item<float>(), b[i].item<float>());
    }
    return 1;
  }
  return 0;
}

// AdaHGComputation parity: 4D in/out wrapper around AdaHGConv.
// We construct the module, then load the same hgconv weights it owns
// internally, and compare against the dumped 4D output.
static int run_hgcomp(const std::string& base, const std::string& tag,
                      const std::string& hg_tag) {
  if (!fs::exists(base + "/" + tag + "_inp.bin")) {
    std::printf("[skip] %s/%s_inp.bin missing\n", base.c_str(), tag.c_str());
    return 0;
  }
  auto inp_ref = read_bin(base + "/" + tag + "_inp.bin", {1, 64, 40, 40});
  auto out_ref = read_bin(base + "/" + tag + "_out.bin", {1, 64, 40, 40});

  auto proto = read_bin(base + "/" + hg_tag +
                        "_edge_generator_prototype_base.bin", {4, 64});
  auto ctx_w = read_bin(base + "/" + hg_tag +
                        "_edge_generator_context_net_weight.bin", {256, 128});
  auto ctx_b = read_bin(base + "/" + hg_tag +
                        "_edge_generator_context_net_bias.bin", {256});
  auto php_w = read_bin(base + "/" + hg_tag +
                        "_edge_generator_pre_head_proj_weight.bin", {64, 64});
  auto php_b = read_bin(base + "/" + hg_tag +
                        "_edge_generator_pre_head_proj_bias.bin", {64});
  auto ep_w  = read_bin(base + "/" + hg_tag + "_edge_proj_0_weight.bin",
                        {64, 64});
  auto ep_b  = read_bin(base + "/" + hg_tag + "_edge_proj_0_bias.bin", {64});
  auto np_w  = read_bin(base + "/" + hg_tag + "_node_proj_0_weight.bin",
                        {64, 64});
  auto np_b  = read_bin(base + "/" + hg_tag + "_node_proj_0_bias.bin", {64});

  yolocpp::models::AdaHGComputation comp(/*embed_dim=*/64,
                                          /*num_hyperedges=*/4,
                                          /*num_heads=*/4,
                                          /*context=*/"both");
  torch::NoGradGuard ng;
  comp->hgnn->edge_generator->prototype_base.copy_(proto);
  comp->hgnn->edge_generator->context_net->weight.copy_(ctx_w);
  comp->hgnn->edge_generator->context_net->bias.copy_(ctx_b);
  comp->hgnn->edge_generator->pre_head_proj->weight.copy_(php_w);
  comp->hgnn->edge_generator->pre_head_proj->bias.copy_(php_b);
  comp->hgnn->edge_proj[0]->as<torch::nn::LinearImpl>()->weight.copy_(ep_w);
  comp->hgnn->edge_proj[0]->as<torch::nn::LinearImpl>()->bias.copy_(ep_b);
  comp->hgnn->node_proj[0]->as<torch::nn::LinearImpl>()->weight.copy_(np_w);
  comp->hgnn->node_proj[0]->as<torch::nn::LinearImpl>()->bias.copy_(np_b);
  comp->eval();

  auto out_ours = comp->forward(inp_ref);
  auto diff = (out_ours - out_ref).abs();
  double max_abs = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  std::printf("[v13/hgcomp] %s: max|Δ|=%.3e mean|Δ|=%.3e\n",
              tag.c_str(), max_abs, mean_abs);
  if (max_abs > 1e-5) {
    std::printf("[fail] %s: max|Δ| > 1e-5\n", tag.c_str());
    return 1;
  }
  return 0;
}

// Generic loader: copy each named param/buffer from <base>/<tag>_<key>.bin
// where <key> is the dot-name with '.' replaced by '_'.
static void load_module_by_dump(torch::nn::Module& mod, const std::string& base,
                                 const std::string& tag) {
  torch::NoGradGuard ng;
  auto load_one = [&](const std::string& key, torch::Tensor t) {
    std::string fname = base + "/" + tag + "_";
    for (char c : key) fname.push_back(c == '.' ? '_' : c);
    fname += ".bin";
    if (!fs::exists(fname)) return;  // num_batches_tracked etc.
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    auto loaded = read_bin(fname, shape);
    t.copy_(loaded);
  };
  for (auto& p : mod.named_parameters())
    load_one(p.key(), p.value());
  for (auto& p : mod.named_buffers())
    load_one(p.key(), p.value());
}

static int run_c3ah(const std::string& base, const std::string& tag) {
  if (!fs::exists(base + "/" + tag + "_inp.bin")) {
    std::printf("[skip] %s/%s_inp.bin missing\n", base.c_str(), tag.c_str());
    return 0;
  }
  auto inp_ref = read_bin(base + "/" + tag + "_inp.bin", {1, 64, 40, 40});
  auto out_ref = read_bin(base + "/" + tag + "_out.bin", {1, 64, 40, 40});

  yolocpp::models::C3AH c(/*c1=*/64, /*c2=*/64, /*e=*/1.0,
                           /*num_hyperedges=*/4, /*context=*/"both");
  load_module_by_dump(*c.ptr(), base, tag);
  c->eval();
  auto out_ours = c->forward(inp_ref);
  auto diff = (out_ours - out_ref).abs();
  double max_abs  = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  std::printf("[v13/c3ah] %s: max|Δ|=%.3e mean|Δ|=%.3e\n",
              tag.c_str(), max_abs, mean_abs);
  if (max_abs > 1e-4) {
    std::printf("[fail] %s: max|Δ| > 1e-4\n", tag.c_str());
    return 1;
  }
  return 0;
}

static int run_hyperace(const std::string& base) {
  const std::string tag = "hyperace";
  if (!fs::exists(base + "/" + tag + "_inp0.bin")) {
    std::printf("[skip] %s/%s_inp0.bin missing\n", base.c_str(), tag.c_str());
    return 0;
  }
  auto inp0 = read_bin(base + "/" + tag + "_inp0.bin", {1, 128, 80, 80});
  auto inp1 = read_bin(base + "/" + tag + "_inp1.bin", {1, 128, 40, 40});
  auto inp2 = read_bin(base + "/" + tag + "_inp2.bin", {1, 256, 20, 20});
  auto out_ref = read_bin(base + "/" + tag + "_out.bin", {1, 128, 40, 40});

  // For v13n: c1 = scaled L6 = 128, c2 = scaled 512 (=128 at width=0.25),
  // n=1 (parse_model override), num_hyperedges = 8 * 0.5 = 4 (n-scale).
  // dsc3k=True, shortcut=True, e1=0.5, e2=1.0, ctx="both",
  // channel_adjust=True (default for n/s).
  yolocpp::models::HyperACE h(
      /*c1=*/128, /*c2=*/128, /*n=*/1, /*num_hyperedges=*/4,
      /*dsc3k=*/true, /*shortcut=*/true, /*e1=*/0.5, /*e2=*/1.0,
      /*ctx=*/"both", /*channel_adjust=*/true);
  load_module_by_dump(*h.ptr(), base, tag);
  h->eval();

  auto out_ours = h->forward({inp0, inp1, inp2});
  auto diff = (out_ours - out_ref).abs();
  double max_abs  = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  std::printf("[v13/hyperace] HyperACE L9: max|Δ|=%.3e mean|Δ|=%.3e\n",
              max_abs, mean_abs);
  if (max_abs > 1e-3) {
    std::printf("[fail] HyperACE: max|Δ| > 1e-3\n");
    auto a = out_ours.flatten();
    auto b = out_ref.flatten();
    for (int i = 0; i < 6; ++i) {
      std::printf("  [%d] ours=%.7f ref=%.7f\n", i,
                  a[i].item<float>(), b[i].item<float>());
    }
    return 1;
  }
  return 0;
}

int main() {
  const std::string base = "/tmp/yolocpp_parity/dumps/yolo13n";
  if (!fs::exists(base + "/ada_b1_inp.bin")) {
    std::printf("[skip] %s/ada_b1_inp.bin missing — Python harness not run\n",
                base.c_str());
    return 0;
  }
  int rc = 0;
  rc |= run_one(base, "b1");
  rc |= run_one(base, "b2");
  rc |= run_hgconv(base, "hgconv_b1");
  rc |= run_hgconv(base, "hgconv_b2");
  rc |= run_hgcomp(base, "hgcomp_b1", "hgconv_b1");
  rc |= run_hgcomp(base, "hgcomp_b2", "hgconv_b2");
  rc |= run_c3ah(base, "c3ah_b1");
  rc |= run_c3ah(base, "c3ah_b2");
  rc |= run_hyperace(base);
  if (rc == 0) std::printf("[pass]\n");
  return rc;
}
