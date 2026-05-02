// Smoke for the public chainable C++ API (#52).
//
// Routes go through the same `cmd_*` bodies the CLI uses, so this
// test mostly verifies the argument-shape adapter and that chaining
// (`.to(...).predict(...)`) compiles + executes.
//
// Gated on `data/yolo11s.pt` being present; if missing, prints SKIP
// and returns 0 so the test doesn't fail on a fresh checkout.

#include <filesystem>
#include <iostream>

#include "yolocpp/api.hpp"

#define EXPECT(cond, msg) \
  do {                    \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  namespace fs = std::filesystem;
  if (!fs::exists("data/yolo11s.pt") || !fs::exists("data/bus.jpg")) {
    std::cout << "[api] SKIP — data/yolo11s.pt or bus.jpg missing\n";
    return 0;
  }

  yolocpp::YOLO m("data/yolo11s.pt");

  // Chaining: `.to(...)` returns YOLO&. Validate device-string check
  // by passing through normalise_device.
  auto& m2 = m.to("auto");
  EXPECT(&m2 == &m, "to() must return *this for chaining");

  // Predict on an image — exercises the registry-routed v11 path
  // through the API. Now returns populated dets (#52A2).
  auto out_path = "/tmp/api_pred.jpg";
  fs::remove(out_path);
  auto dets = m.predict({.source = "data/bus.jpg", .out = out_path});
  EXPECT(fs::exists(out_path), "predict didn't produce output jpg");
  EXPECT(!dets.empty(), "predict returned no detections (expected ≥1 on bus.jpg)");
  // bus.jpg with yolo11s @ default conf=0.25 produces 5 dets — be
  // tolerant of conf-thresh drift across versions, just require a
  // reasonable lower bound.
  EXPECT(dets.size() >= 3, "predict returned suspiciously few detections");
  // Sanity-check the first det shape: bbox in pixel coords with
  // x1<x2 and y1<y2, conf in [0,1], non-negative class.
  const auto& d0 = dets[0];
  EXPECT(d0.x2 > d0.x1 && d0.y2 > d0.y1, "det bbox malformed");
  EXPECT(d0.conf >= 0.0f && d0.conf <= 1.0f, "det conf out of range");
  EXPECT(d0.cls >= 0, "det class index negative");
  fs::remove(out_path);

  // Predict on a directory — exercises the dir/glob fan-out via the
  // API. Single-input dir keeps the CLI's "out becomes file" rule;
  // we use a multi-image dir to force the directory output mode.
  fs::create_directories("/tmp/api_srcdir");
  fs::copy_file("data/bus.jpg", "/tmp/api_srcdir/a.jpg",
                fs::copy_options::overwrite_existing);
  fs::copy_file("data/bus.jpg", "/tmp/api_srcdir/b.jpg",
                fs::copy_options::overwrite_existing);
  m.predict({.source = "/tmp/api_srcdir", .out = "/tmp/api_outdir"});
  EXPECT(fs::exists("/tmp/api_outdir/a_detect.jpg"), "dir fan-out a missing");
  EXPECT(fs::exists("/tmp/api_outdir/b_detect.jpg"), "dir fan-out b missing");
  fs::remove_all("/tmp/api_srcdir");
  fs::remove_all("/tmp/api_outdir");

  // Export — exercises the export routing.
  fs::remove("/tmp/api_export.onnx");
  m.export_({.format = "onnx", .out = "/tmp/api_export.onnx",
             .precision = "fp16"});
  EXPECT(fs::exists("/tmp/api_export.onnx"), "export didn't produce onnx");
  EXPECT(fs::file_size("/tmp/api_export.onnx") > 1'000'000,
         "exported onnx is suspiciously small");
  fs::remove("/tmp/api_export.onnx");

  // Bad precision should throw.
  bool threw = false;
  try {
    m.export_({.format = "onnx", .precision = "garbage"});
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT(threw, "export with bogus precision should throw");

  std::cout << "[api] OK — predict (image + dir), export, chaining\n";
  return 0;
}
