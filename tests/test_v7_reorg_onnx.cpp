// Regression test for the v7 ReOrg ONNX emitter channel-order bug.
//
// The v7 model's ReOrg layer is `torch::pixel_unshuffle(x, 2)`, whose output
// channel index is `c*4 + (sy*2 + sx)` — the ORIGINAL channel `c` is the
// slowest-varying group ("CRD" order). ONNX `SpaceToDepth` is DCR-only (block
// index slowest: `(sy*2 + sx)*C + c`), so emitting a bare SpaceToDepth
// channel-permutes the output and scrambles every downstream conv. The emitter
// now follows SpaceToDepth with a reshape → transpose → reshape that regroups
// DCR → CRD.
//
// We can't run the full v7 decode graph through cv::dnn (OpenCV 4.6 chokes on
// the decode's 5-D ops; TRT — the real target — handles them), and no
// onnxruntime is available here. So we prove the emitted op sequence directly:
// replicate each emitted ONNX node with its documented semantics in LibTorch
// and assert the result equals pixel_unshuffle — and that a bare SpaceToDepth
// does NOT (i.e. the bug was real). The ops involved (SpaceToDepth / Reshape /
// Transpose) are standard and faithfully implemented by every ONNX runtime;
// the only thing in question is the channel order, which this pins down.
//
// A structural check additionally confirms a real v7-w6 model (which starts
// with ReOrg) exports to a well-formed ONNX through the actual emitter.

#include <torch/torch.h>

#include <cstdio>
#include <filesystem>
#include <iostream>

#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/serialization/onnx_export.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; }  \
  } while (0)

namespace {

// ONNX `SpaceToDepth(blocksize=bs)` — DCR semantics, per the operator spec.
torch::Tensor onnx_space_to_depth_dcr(const torch::Tensor& x, int bs) {
  auto b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
  auto t = x.reshape({b, c, h / bs, bs, w / bs, bs});
  t = t.permute({0, 3, 5, 1, 2, 4}).contiguous();  // block dims outer
  return t.reshape({b, c * bs * bs, h / bs, w / bs});
}

// The emitted regroup: Reshape(B, bs², C, H', W') → Transpose(0,2,1,3,4)
// → Reshape(B, C·bs², H', W'). Turns DCR (block·C + c) into CRD (c·bs² + block).
torch::Tensor regroup_dcr_to_crd(const torch::Tensor& sd, int c, int bs) {
  auto b = sd.size(0), hp = sd.size(2), wp = sd.size(3);
  auto r = sd.reshape({b, (int64_t)(bs * bs), (int64_t)c, hp, wp});
  r = r.permute({0, 2, 1, 3, 4}).contiguous();
  return r.reshape({b, (int64_t)(c * bs * bs), hp, wp});
}

}  // namespace

int main() {
  using namespace yolocpp;
  torch::manual_seed(0);
  const int bs = 2;

  // 1) Channel-order proof on a small tensor with distinct values.
  {
    const int C = 3, H = 8, W = 8;
    auto x  = torch::rand({2, C, H, W});
    auto pu = torch::pixel_unshuffle(x, bs);          // the model's ReOrg

    auto sd       = onnx_space_to_depth_dcr(x, bs);   // bare SpaceToDepth (old)
    auto emitted  = regroup_dcr_to_crd(sd, C, bs);    // SpaceToDepth + regroup (new)

    EXPECT(sd.sizes() == pu.sizes(),
           "SpaceToDepth and pixel_unshuffle agree on SHAPE");
    EXPECT(!torch::allclose(sd, pu),
           "bare SpaceToDepth (DCR) does NOT match pixel_unshuffle — bug was real");
    EXPECT(torch::allclose(emitted, pu, 1e-6, 1e-6),
           "SpaceToDepth + DCR→CRD regroup == pixel_unshuffle");
    std::cerr << "[v7-reorg] channel-order proof OK "
              << "(max|Δ| emitted-vs-pu = "
              << (emitted - pu).abs().max().item<float>() << ")\n";
  }

  // 2) Structural: a real v7-w6 model (starts with ReOrg) exports cleanly.
  {
    models::Yolo7 m(models::Yolo7Scale::W6, /*nc=*/80);
    m->eval();
    // A forward pass initialises lazy stride/anchor state the exporter reads.
    { torch::NoGradGuard ng; m->forward_eval(torch::rand({1, 3, 640, 640})); }
    serialization::OnnxExportConfig cfg; cfg.imgsz = 640;
    const std::string onnx = "build/v7_w6_reorg.onnx";
    fs::remove(onnx);
    serialization::export_yolo7_onnx(m, onnx, cfg);
    EXPECT(fs::exists(onnx), "v7-w6 ONNX not written");
    EXPECT(fs::file_size(onnx) > 1'000'000u,
           "v7-w6 ONNX suspiciously small (< 1 MB)");
    std::cerr << "[v7-reorg] v7-w6 structural export OK ("
              << fs::file_size(onnx) << " bytes)\n";
    fs::remove(onnx);  // don't leave the large artifact around
  }

  std::cout << "ALL PASS\n";
  return 0;
}
