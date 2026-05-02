// #65L slice 14 — standalone probe isolating top-K + refpoint_embed
// subset against Python references.
//
// Earlier slices showed encoder-output stages match bit-exactly,
// but refpoints fed to sineembed differ → tracing the divergence
// to either:
//   (a) `top-K idx` selecting different queries on near-equal cls
//       scores (libtorch vs PyTorch tie-breaking).
//   (b) `refpoint_embed[:Q]` slice having wrong values (loader bug).
//
// This binary loads only the model + the relevant Python dumps
// and compares each candidate in isolation, avoiding the
// memory-lifetime issues that crashed the combined harness.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/models/rfdetr_transformer.hpp"
#include "yolocpp/serialization/rfdetr_weights.hpp"

namespace fs = std::filesystem;

namespace {

torch::Tensor load_dump(const fs::path& dir, const std::string& name) {
  fs::path bin = dir / (name + ".bin");
  fs::path shape_p = dir / (name + ".shape");
  if (!fs::exists(bin) || !fs::exists(shape_p)) return {};
  std::ifstream f_shape(shape_p);
  std::stringstream ss; ss << f_shape.rdbuf();
  std::vector<int64_t> shape;
  int64_t v; while (ss >> v) shape.push_back(v);
  int64_t numel = 1;
  for (auto s : shape) numel *= s;
  std::vector<float> buf(numel);
  std::ifstream f_bin(bin, std::ios::binary);
  f_bin.read(reinterpret_cast<char*>(buf.data()), numel * sizeof(float));
  return torch::from_blob(buf.data(), shape, torch::kFloat).clone();
}

}  // namespace

int main() {
  const char* dir_p = std::getenv("RFDETR_PARITY_DUMP_DIR");
  const char* w_p   = std::getenv("RFDETR_TEST_WEIGHTS_DIR");
  if (!dir_p || !w_p) {
    std::cout << "SKIP: env vars not set\n";
    return 0;
  }
  fs::path dir(dir_p);
  fs::path wpath = fs::path(w_p) / "rf-detr-base.pth";
  if (!fs::exists(wpath)) { std::cout << "SKIP: weights missing\n"; return 0; }

  yolocpp::models::RFDetr m(yolocpp::models::kRfdetrBase, /*nc=*/80);
  m->eval();
  m->load_from_upstream_pt(wpath.string(), /*strict=*/false);

  torch::NoGradGuard ng;

  // Probe 1: refpoint_embed[:Q] slice.
  {
    int Q = m->scale.num_queries;
    auto cpp_rpe = m->named_parameters()["refpoint_embed.weight"]
                      .slice(0, 0, Q).contiguous();
    auto py_rpe  = load_dump(dir, "refpoint_embed_weight");
    if (py_rpe.defined()) {
      auto diff = (cpp_rpe - py_rpe).abs().max().item<float>();
      std::cout << "[probe-A] refpoint_embed[:Q]  shape=" << cpp_rpe.sizes()
                << "  cpp_sum=" << cpp_rpe.sum().item<float>()
                << "  py_sum="  << py_rpe.sum().item<float>()
                << "  max_abs_diff=" << diff << "\n";
    }
  }

  // Probe 2: top-K selection. Recreate the inputs from scratch.
  {
    torch::manual_seed(42);
    auto inp = torch::randn({1, 3, 560, 560});

    auto& slot_inner = *m->named_children()["backbone"]
                          ->as<torch::nn::ModuleListImpl>()
                          ->ptr(0)
                          ->as<yolocpp::models::rfdetr::BackboneSlotImpl>();
    auto memory_2d = slot_inner.forward(inp);
    auto memory = memory_2d.flatten(2).transpose(1, 2).contiguous();
    auto props = yolocpp::models::rfdetr::gen_encoder_output_proposals_1l(
        memory, memory_2d.size(2), memory_2d.size(3));
    auto& trans = *m->named_modules()["transformer"]
                       ->as<yolocpp::models::rfdetr::RFDetrTransformerImpl>();
    auto out_mem = trans.enc_output_norm[0]->as<torch::nn::LayerNormImpl>()
                       ->forward(trans.enc_output[0]->as<torch::nn::LinearImpl>()
                                       ->forward(props.output_memory));
    auto cls_l = trans.enc_out_class_embed[0]->as<torch::nn::LinearImpl>()
                       ->forward(out_mem);

    auto top_scores = std::get<0>(cls_l.max(-1));     // [1, L]
    int  K = m->scale.num_queries;
    auto sorted = torch::sort(top_scores, /*stable=*/true,
                                /*dim=*/1, /*descending=*/true);
    auto cpp_idx = std::get<1>(sorted).slice(1, 0, K).contiguous();
    auto cpp_top_scores = top_scores.gather(1, cpp_idx).contiguous();

    auto py_boxes_ts = load_dump(dir, "boxes_ts");    // python topk refpts
    if (py_boxes_ts.defined()) {
      // Recompute coord, gather at cpp's topk, compare to python's boxes_ts.
      auto bbox_d = trans.enc_out_bbox_embed[0]->as<yolocpp::models::rfdetr::RFDetrMLPImpl>()
                       ->forward(out_mem);
      auto cx = bbox_d.slice(-1, 0, 2) * props.output_proposals.slice(-1, 2, 4) +
                  props.output_proposals.slice(-1, 0, 2);
      auto wh = bbox_d.slice(-1, 2, 4).exp() * props.output_proposals.slice(-1, 2, 4);
      auto coord = torch::cat({cx, wh}, -1);
      auto idx4 = cpp_idx.unsqueeze(-1).expand({1, K, 4});
      auto cpp_boxes_ts = torch::gather(coord, 1, idx4).contiguous();
      std::cout << "[probe-B] boxes_ts  cpp_sum=" << cpp_boxes_ts.sum().item<float>()
                << "  py_sum=" << py_boxes_ts.sum().item<float>()
                << "  max_abs_diff="
                << (cpp_boxes_ts - py_boxes_ts).abs().max().item<float>() << "\n";
    }

    // Compare top scores per index — useful to see if scores match
    // exactly even if topk-pick orders differ.
    // Compare full topk_idx + scores against Python.
    auto py_idx = load_dump(dir, "topk_first_idx");
    if (py_idx.defined()) {
      // Python saved as int64 fp32-converted; reload as int64.
      fs::path bin = dir / "topk_first_idx.bin";
      std::ifstream f(bin, std::ios::binary);
      std::vector<int64_t> ibuf(K);
      f.read(reinterpret_cast<char*>(ibuf.data()), K * sizeof(int64_t));
      auto py_idx_t = torch::from_blob(ibuf.data(), {1, K}, torch::kLong).clone();
      int n_diff = 0, first_diff = -1;
      for (int i = 0; i < K; ++i) {
        if (cpp_idx[0][i].item<int64_t>() != py_idx_t[0][i].item<int64_t>()) {
          n_diff++;
          if (first_diff < 0) first_diff = i;
        }
      }
      std::cout << "[probe-B] topk_idx differences: " << n_diff << "/" << K
                << "  first_diff_at=" << first_diff << "\n";
      if (first_diff >= 0) {
        auto ci = cpp_idx[0][first_diff].item<int64_t>();
        auto pi = py_idx_t[0][first_diff].item<int64_t>();
        std::cout.precision(12);
        std::cout << "[probe-B]  at " << first_diff << ": cpp_idx=" << ci
                  << " py_idx=" << pi
                  << "  score@cpp_idx=" << top_scores[0][ci].item<float>()
                  << "  score@py_idx="  << top_scores[0][pi].item<float>() << "\n";
        // Print as raw uint32 bits to catch sub-fp32-display differences.
        union { float f; uint32_t u; } a, b;
        a.f = top_scores[0][ci].item<float>();
        b.f = top_scores[0][pi].item<float>();
        std::cout << "[probe-B]  bits: cpp=0x" << std::hex << a.u
                  << " py=0x" << b.u << std::dec << "\n";
      }
    }
  }

  return 0;
}
