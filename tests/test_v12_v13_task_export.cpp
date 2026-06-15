// Data-free structural smoke for v12/v13 task ONNX export (segment / pose /
// obb / classify). No upstream task weights ship for v12/v13, so we build each
// head from random init, run a forward to confirm the architecture is sound,
// emit the ONNX graph, and assert the file is written with the expected output
// tensor names present in the serialized protobuf. This guards the new walk_v12
// / walk_v13 trunks + the 8 task emitters + the head-index wiring; it always
// runs (no weights/data needed).

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yolocpp/models/yolo12_tasks.hpp"
#include "yolocpp/models/yolo13_tasks.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_save.hpp"

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; }  \
  } while (0)

namespace {

namespace fs = std::filesystem;
namespace ser = yolocpp::serialization;

// Read a file into a string (binary) so we can grep the protobuf for the
// expected output tensor names (GraphBuilder writes them as raw strings).
std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool has(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  torch::manual_seed(0);
  torch::NoGradGuard ng;
  fs::create_directories("build/onnx_task_smoke");
  auto out = [](const std::string& s) {
    return std::string("build/onnx_task_smoke/") + s;
  };
  // 256² keeps the random-init forward + export cheap while exercising all
  // three detect levels (strides 8/16/32).
  ser::OnnxExportConfig cfg;
  cfg.imgsz = 256;
  auto x = torch::randn({1, 3, cfg.imgsz, cfg.imgsz});

  using namespace yolocpp::models;

  // ─── v12 ────────────────────────────────────────────────────────────────
  {
    auto sc = yolo12_scale_from_letter("n");
    {
      Yolo12Segment m(sc, /*nc=*/80); m->eval();
      auto [d, c, p] = m->forward_eval(x);
      EXPECT(d.size(0) == 1 && d.size(1) == 4 + 80, "v12-seg forward shape");
      std::string path = out("yolo12n-seg.onnx");
      ser::export_yolo12_segment_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024, "v12-seg onnx non-trivial");
      EXPECT(has(b, "output") && has(b, "coefs") && has(b, "protos"),
             "v12-seg onnx output names");
      // Persist a loadable checkpoint so the registry/CLI export + TRT-parse
      // path can be exercised against a real .pt (build/ is gitignored).
      ser::save_module_state_dict(*m, out("yolo12n-seg.pt"));
      std::cout << "[v12-seg] ok (" << b.size() << " B)\n";
    }
    {
      Yolo12Pose m(sc, /*nc=*/1, 17, 3); m->eval();
      auto [d, k] = m->forward_eval(x);
      EXPECT(d.size(1) == 4 + 1, "v12-pose forward shape");
      std::string path = out("yolo12n-pose.onnx");
      ser::export_yolo12_pose_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "keypoints"), "v12-pose onnx");
      std::cout << "[v12-pose] ok (" << b.size() << " B)\n";
    }
    {
      Yolo12OBB m(sc, /*nc=*/15, /*ne=*/1); m->eval();
      auto [d, a] = m->forward_eval(x);
      EXPECT(d.size(1) == 4 + 15, "v12-obb forward shape");
      std::string path = out("yolo12n-obb.onnx");
      ser::export_yolo12_obb_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "angle"), "v12-obb onnx");
      std::cout << "[v12-obb] ok (" << b.size() << " B)\n";
    }
    {
      Yolo12Classify m(sc, /*nc=*/1000); m->eval();
      auto y = m->forward(torch::randn({1, 3, 224, 224}));
      EXPECT(y.size(1) == 1000, "v12-cls forward shape");
      ser::OnnxExportConfig ccfg; ccfg.imgsz = 224;
      std::string path = out("yolo12n-cls.onnx");
      ser::export_yolo12_classify_onnx(m, path, ccfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "output"), "v12-cls onnx");
      std::cout << "[v12-cls] ok (" << b.size() << " B)\n";
    }
  }

  // ─── v13 ────────────────────────────────────────────────────────────────
  {
    auto sc = yolo13_scale_from_letter("n");
    {
      Yolo13Segment m(sc, /*nc=*/80); m->eval();
      auto [d, c, p] = m->forward_eval(x);
      EXPECT(d.size(1) == 4 + 80, "v13-seg forward shape");
      std::string path = out("yolo13n-seg.onnx");
      ser::export_yolo13_segment_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024, "v13-seg onnx non-trivial");
      EXPECT(has(b, "output") && has(b, "coefs") && has(b, "protos"),
             "v13-seg onnx output names");
      ser::save_module_state_dict(*m, out("yolo13n-seg.pt"));
      std::cout << "[v13-seg] ok (" << b.size() << " B)\n";
    }
    {
      Yolo13Pose m(sc, /*nc=*/1, 17, 3); m->eval();
      auto [d, k] = m->forward_eval(x);
      EXPECT(d.size(1) == 4 + 1, "v13-pose forward shape");
      std::string path = out("yolo13n-pose.onnx");
      ser::export_yolo13_pose_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "keypoints"), "v13-pose onnx");
      std::cout << "[v13-pose] ok (" << b.size() << " B)\n";
    }
    {
      Yolo13OBB m(sc, /*nc=*/15, /*ne=*/1); m->eval();
      auto [d, a] = m->forward_eval(x);
      EXPECT(d.size(1) == 4 + 15, "v13-obb forward shape");
      std::string path = out("yolo13n-obb.onnx");
      ser::export_yolo13_obb_onnx(m, path, cfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "angle"), "v13-obb onnx");
      std::cout << "[v13-obb] ok (" << b.size() << " B)\n";
    }
    {
      Yolo13Classify m(sc, /*nc=*/1000); m->eval();
      auto y = m->forward(torch::randn({1, 3, 224, 224}));
      EXPECT(y.size(1) == 1000, "v13-cls forward shape");
      ser::OnnxExportConfig ccfg; ccfg.imgsz = 224;
      std::string path = out("yolo13n-cls.onnx");
      ser::export_yolo13_classify_onnx(m, path, ccfg);
      auto b = slurp(path);
      EXPECT(b.size() > 1024 && has(b, "output"), "v13-cls onnx");
      std::cout << "[v13-cls] ok (" << b.size() << " B)\n";
    }
  }

  std::cout << "=== v12/v13 task ONNX export smoke PASS ===\n";
  return 0;
}
