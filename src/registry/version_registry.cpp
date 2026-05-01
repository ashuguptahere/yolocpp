// Per-version registry — see include/yolocpp/registry/version_adapter.hpp
// for the contract and the "how to add a new YOLO version" walkthrough.
//
// This TU owns:
//   1. The Registry singleton (idempotent register/find).
//   2. Per-version registration helpers (`register_yolo<N>`).
//   3. `register_all_versions()` — the one entry point CLI / API call.
//
// First migration target: ONNX export (clearest concentration of
// per-version dispatch, ~250 lines in cmd_export). Predict / val /
// train will follow in subsequent tasks (#46D, #46E, #46F).

#include "yolocpp/registry/version_adapter.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include <torch/torch.h>

#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo26_tasks.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace yolocpp::registry {

Registry& Registry::instance() {
  static Registry r;
  return r;
}

void Registry::register_version(VersionAdapter a) {
  versions_[a.version_id] = std::move(a);
}

const VersionAdapter* Registry::find(const std::string& version_id) const {
  auto it = versions_.find(version_id);
  return it == versions_.end() ? nullptr : &it->second;
}

std::vector<std::string> Registry::known_ids() const {
  std::vector<std::string> ids;
  ids.reserve(versions_.size());
  for (const auto& kv : versions_) ids.push_back(kv.first);
  std::sort(ids.begin(), ids.end());
  return ids;
}

namespace {

// Helpers — keep the per-version blocks below readable.

void load_and_eval(auto& m, const std::string& weights, int imgsz,
                   bool needs_warmup) {
  auto sd = serialization::load_state_dict(weights);
  m->load_from_state_dict(sd.entries);
  if (needs_warmup) {
    torch::NoGradGuard ng;
    auto x = torch::zeros({1, 3, imgsz, imgsz});
    m->forward_eval(x);  // populates strides for v3/v7/v9/v10
  }
  m->eval();
}

// v6 has the most complex scale/variant matrix — pull it out so the
// adapter lambda stays tight.
struct V6Resolved {
  models::Yolo6Variant variant_kind = models::Yolo6Variant::Standard;
  models::Yolo6Scale scale = models::kYolo6s;
  bool p6 = false;
};

V6Resolved resolve_v6(const std::string& s) {
  V6Resolved r;
  if      (s == "n")        r.scale = models::kYolo6n;
  else if (s == "s" || s.empty()) r.scale = models::kYolo6s;
  else if (s == "m")        r.scale = models::kYolo6m;
  else if (s == "l")        r.scale = models::kYolo6l;
  else if (s == "n6")       { r.scale = models::kYolo6n; r.p6 = true; }
  else if (s == "s6")       { r.scale = models::kYolo6s; r.p6 = true; }
  else if (s == "m6")       { r.scale = models::kYolo6m; r.p6 = true; }
  else if (s == "l6")       { r.scale = models::kYolo6l; r.p6 = true; }
  else if (s == "s_mbla")   r.scale = models::kYolo6s_mbla;
  else if (s == "m_mbla")   r.scale = models::kYolo6m_mbla;
  else if (s == "l_mbla")   r.scale = models::kYolo6l_mbla;
  else if (s == "x_mbla")   r.scale = models::kYolo6x_mbla;
  return r;
}

// v8's scale-letter helper lives in cli/main.cpp; inline a copy here so
// the registry doesn't depend on CLI internals. Mirrors the v8.hpp
// constexpr scale enum (kYolo8n / s / m / l / x).
models::Yolo8Scale v8_scale_from_letter(const std::string& s) {
  if (s == "n") return models::kYolo8n;
  if (s == "s" || s.empty()) return models::kYolo8s;
  if (s == "m") return models::kYolo8m;
  if (s == "l") return models::kYolo8l;
  if (s == "x") return models::kYolo8x;
  return models::kYolo8n;
}

// Default imgsz lookups shared by multiple versions.
int detect_imgsz_default(const std::string& /*scale*/,
                         const std::string& task) {
  if (task == "classify") return 224;
  return 0;  // 0 ⇒ caller default (640)
}

// ---- v3 ------------------------------------------------------------------
VersionAdapter make_v3() {
  VersionAdapter a;
  a.version_id = "v3";
  a.display_name = "yolo3";
  a.upstream_year = "2018";
  a.default_export_basename = "yolo3";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string&,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo3 export: only 'detect' supported");
    models::Yolo3 m(models::kYolo3, nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo3_onnx(m, path, cfg);
  };
  return a;
}

// ---- v4 ------------------------------------------------------------------
VersionAdapter make_v4() {
  VersionAdapter a;
  a.version_id = "v4";
  a.display_name = "yolo4";
  a.upstream_year = "2020";
  a.default_export_basename = "yolo4";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string&, const std::string& task) {
    if (task == "classify") return 224;
    return 608;  // v4 anchors are calibrated to 608²
  };
  a.export_onnx = [](const std::string& weights, const std::string&,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo4 export: only 'detect' supported");
    models::Yolo4 m(nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo4_onnx(m, path, cfg);
  };
  return a;
}

// ---- v5 ------------------------------------------------------------------
VersionAdapter make_v5() {
  VersionAdapter a;
  a.version_id = "v5";
  a.display_name = "yolo5";
  a.upstream_year = "2020";
  a.default_export_basename = "yolo5";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo5 export: only 'detect' supported");
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo5_onnx(m, path, cfg);
  };
  return a;
}

// ---- v6 ------------------------------------------------------------------
VersionAdapter make_v6() {
  VersionAdapter a;
  a.version_id = "v6";
  a.display_name = "yolo6";
  a.upstream_year = "2022";
  a.default_export_basename = "yolo6";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string& scale, const std::string& task) {
    if (task == "classify") return 224;
    if (scale == "n6" || scale == "s6" || scale == "m6" || scale == "l6")
      return 1280;  // P6 variants
    return 0;
  };
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo6 export: only 'detect' supported");
    auto r = resolve_v6(scale);
    models::Yolo6 m(nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo6_onnx(m, path, cfg);
  };
  return a;
}

// ---- v7 ------------------------------------------------------------------
VersionAdapter make_v7() {
  VersionAdapter a;
  a.version_id = "v7";
  a.display_name = "yolo7";
  a.upstream_year = "2022";
  a.default_export_basename = "yolo7";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string& scale, const std::string& task) {
    if (task == "classify") return 224;
    if (scale == "w6" || scale == "e6" || scale == "d6" || scale == "e6e")
      return 1280;  // P6 variants
    return 0;
  };
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo7 export: only 'detect' supported");
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo7_onnx(m, path, cfg);
  };
  return a;
}

// ---- v8 (full task family) ----------------------------------------------
VersionAdapter make_v8() {
  VersionAdapter a;
  a.version_id = "v8";
  a.display_name = "yolo8";
  a.upstream_year = "2023";
  a.default_export_basename = "yolo8";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = v8_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo8Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo8Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo8Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo8Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo8OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo8 export: unknown task '" + task + "'");
    }
  };
  return a;
}

// ---- v9 ------------------------------------------------------------------
VersionAdapter make_v9() {
  VersionAdapter a;
  a.version_id = "v9";
  a.display_name = "yolo9";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo9";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo9 export: only 'detect' supported");
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo9_onnx(m, path, cfg);
  };
  return a;
}

// ---- v10 -----------------------------------------------------------------
VersionAdapter make_v10() {
  VersionAdapter a;
  a.version_id = "v10";
  a.display_name = "yolo10";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo10";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.trt_disable_tf32 = true;  // RepVGGDW saturation across all scales
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo10 export: only 'detect' supported");
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo10_onnx(m, path, cfg);
  };
  return a;
}

// ---- v11 (full task family) ---------------------------------------------
VersionAdapter make_v11() {
  VersionAdapter a;
  a.version_id = "v11";
  a.display_name = "yolo11";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo11";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = models::yolo11_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo11Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo11Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo11Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo11Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo11OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo11 export: unknown task '" + task + "'");
    }
  };
  return a;
}

// ---- v12 -----------------------------------------------------------------
VersionAdapter make_v12() {
  VersionAdapter a;
  a.version_id = "v12";
  a.display_name = "yolo12";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo12";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error(
          "yolo12 export: only 'detect' supported (task heads scaffolded; "
          "weights pending — see TODO #60)");
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo12_onnx(m, path, cfg);
  };
  return a;
}

// ---- v13 -----------------------------------------------------------------
VersionAdapter make_v13() {
  VersionAdapter a;
  a.version_id = "v13";
  a.display_name = "yolo13";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo13";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error(
          "yolo13 export: only 'detect' supported (no `m` scale upstream)");
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo13_onnx(m, path, cfg);
  };
  return a;
}

// ---- v26 (full task family) ---------------------------------------------
VersionAdapter make_v26() {
  VersionAdapter a;
  a.version_id = "v26";
  a.display_name = "yolo26";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo26";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = models::yolo26_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo26Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo26Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo26Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo26Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo26OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo26 export: unknown task '" + task + "'");
    }
  };
  return a;
}

}  // namespace

void register_all_versions() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto& r = Registry::instance();
    r.register_version(make_v3());
    r.register_version(make_v4());
    r.register_version(make_v5());
    r.register_version(make_v6());
    r.register_version(make_v7());
    r.register_version(make_v8());
    r.register_version(make_v9());
    r.register_version(make_v10());
    r.register_version(make_v11());
    r.register_version(make_v12());
    r.register_version(make_v13());
    r.register_version(make_v26());
  });
}

}  // namespace yolocpp::registry
