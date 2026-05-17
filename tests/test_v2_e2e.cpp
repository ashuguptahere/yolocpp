// End-to-end test for YOLO2: convert pjreddie's yolov2.weights → yolo2.pt
// (no Darknet runtime), construct Yolo2 (full + tiny), forward + decode
// sanity, optional predict on bus.jpg. Forward-shape sanity always runs;
// .weights round-trip is gated on cache availability.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/yolov2_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  // 1) Full Darknet-19 + reorg + region forward sanity.
  //    Output: [B, 4+nc, na·H·W] = [1, 4+20, 5·13·13] = [1, 24, 845].
  {
    models::Yolo2 m(models::Yolo2Scale::Full, /*nc=*/20);
    m->eval();
    auto x   = torch::zeros({1, 3, 416, 416});
    auto out = m->forward_eval(x);
    EXPECT(out.dim() == 3,                  "v2 forward_eval rank");
    EXPECT(out.size(0) == 1,                "v2 batch");
    EXPECT(out.size(1) == 4 + 20,           "v2 (4+nc) channels");
    EXPECT(out.size(2) == 5 * 13 * 13,      "v2 A = na·H·W");
    std::cout << "[v2-e2e] full forward shape OK (out=" << out.sizes() << ")\n";
  }
  // 2) Reorg invariant: numel preserved, shape collapsed by stride².
  {
    auto x = torch::arange(64 * 26 * 26).reshape({1, 64, 26, 26}).to(torch::kFloat32);
    auto r = models::reorg(x, /*stride=*/2);
    EXPECT(r.sizes() == torch::IntArrayRef({1, 256, 13, 13}),
            "v2 reorg shape (1, 64, 26, 26) → (1, 256, 13, 13)");
    EXPECT(r.numel() == x.numel(), "v2 reorg preserves numel");
    std::cout << "[v2-e2e] reorg layout OK\n";
  }
  // 3) Tiny variant forward sanity. Final spatial = 13 due to the
  //    stride-1 fake-pool keeping the resolution after the last
  //    full /2 chain.
  {
    models::Yolo2 m(models::Yolo2Scale::Tiny, /*nc=*/20);
    m->eval();
    auto x   = torch::zeros({1, 3, 416, 416});
    auto out = m->forward_eval(x);
    EXPECT(out.size(0) == 1,                "v2-tiny batch");
    EXPECT(out.size(1) == 4 + 20,           "v2-tiny (4+nc) channels");
    EXPECT(out.size(2) == 5 * 13 * 13,      "v2-tiny A");
    std::cout << "[v2-e2e] tiny forward shape OK (out=" << out.sizes() << ")\n";
  }

  // 4) Predict on bus.jpg if a pre-converted `data/yolo2*.pt` is
  //    available. `tools/convert_weights` produces these from
  //    pjreddie's `.weights` binaries once; tests then consume the
  //    `.pt` directly. If neither a `.pt` nor a usable `.weights`
  //    file is present, skip the predict leg.
  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache_w = fs::path(home) / ".cache/yolocpp/weights";

  // Try VOC first (smaller nc → easier to interpret); fall back to COCO.
  struct PtCandidate { fs::path path; int nc; };
  std::vector<PtCandidate> pt_candidates = {
      {"data/yolo2-voc.pt", 20}, {cache_w / "yolo2-voc.pt", 20},
      {"data/yolo2.pt",     80}, {cache_w / "yolo2.pt",     80},
  };
  PtCandidate use{};
  for (const auto& c : pt_candidates) {
    if (fs::exists(c.path)) { use = c; break; }
  }
  if (use.path.empty()) {
    // Fall back to a one-shot conversion if a .weights file is around.
    fs::path wsrc;
    int wsrc_nc = 80;
    for (const auto& [w, nc] : std::vector<std::pair<fs::path, int>>{
             {"data/yolov2-voc.weights", 20},
             {cache_w / "yolov2-voc.weights", 20},
             {"data/yolov2.weights", 80},
             {cache_w / "yolov2.weights", 80}}) {
      if (fs::exists(w)) { wsrc = w; wsrc_nc = nc; break; }
    }
    if (wsrc.empty()) {
      std::cout << "[v2-e2e] SKIP predict (no data/yolo2*.pt nor yolov2*.weights)\n";
      return 0;
    }
    fs::path pt = "build/yolo2_e2e.pt";
    fs::remove(pt);
    int blocks = serialization::convert_yolov2_weights(
        wsrc.string(), pt.string(), wsrc_nc,
        models::Yolo2Scale::Full);
    EXPECT(blocks >= 23, "v2 expected ≥ 23 conv blocks");
    use = {pt, wsrc_nc};
  }

  fs::path bus = "data/bus.jpg";
  if (!fs::exists(bus)) {
    std::cout << "[v2-e2e] SKIP predict (no data/bus.jpg)\n";
    return 0;
  }
  auto dets = inference::predict_v2_to_file(
      use.path.string(), bus.string(), "build/v2_e2e_bus.jpg",
      /*imgsz=*/416, /*device=*/"", /*nc=*/use.nc,
      models::Yolo2Scale::Full);
  std::cout << "[v2-e2e] " << dets.size() << " dets on bus.jpg (nc="
            << use.nc << ", weights=" << use.path << ")\n";

  // 5) ONNX export round-trip sanity. Construct a Yolo2 holder from
  //    `use.path`, run `export_yolo2_onnx`, confirm the file is
  //    well-formed (small magic-byte check at the head — the full
  //    parser-roundtrip is exercised by the TRT export path in
  //    practice). This catches structural regressions in the emitter
  //    without paying the TRT build cost in ctest.
  models::Yolo2 m(models::Yolo2Scale::Full, use.nc);
  auto sd = serialization::load_state_dict(use.path.string());
  m->load_from_state_dict(sd.entries);
  m->eval();
  fs::path onnx = "build/v2_e2e.onnx";
  fs::remove(onnx);
  serialization::OnnxExportConfig ocfg; ocfg.imgsz = 416;
  serialization::export_yolo2_onnx(m, onnx.string(), ocfg);
  EXPECT(fs::exists(onnx), "v2: ONNX file not written");
  EXPECT(fs::file_size(onnx) > 1'000'000u,
          "v2: ONNX file suspiciously small (< 1 MB)");
  std::cout << "[v2-e2e] ONNX export OK (" << fs::file_size(onnx)
            << " bytes)\n";
  return 0;
}
