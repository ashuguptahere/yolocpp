// Per-layer parity comparator vs the upstream Python forward.
//
// Reads <dump_dir>/<idx>.bin (raw float32 written by /tmp/yolocpp_parity/
// dump.py — Python harness lives outside the repo by request) and runs the
// same fixed deterministic input through our C++ model layer-by-layer,
// reporting max-abs-diff per YAML index. The first index where the
// difference exceeds a threshold is the bug.
//
// Fixed input: arange(N, float32) / 1000.0  reshaped to [1, 3, imgsz, imgsz]
// (matches dump.py exactly).
//
// Usage:
//   parity_compare <version> <weights.pt> <dump_dir> [imgsz=640]
// where <version> is one of: v11, v26.
//
// This is a dev-only test — registered in CMake but not in ctest because
// it needs the Python dump dir to be present.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "yolocpp/cli/model_info.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace M = yolocpp::models;

namespace {

torch::Tensor fixed_input(int imgsz, torch::Device dev) {
  long long n = 3LL * imgsz * imgsz;
  auto x = torch::arange(n, torch::TensorOptions().dtype(torch::kFloat32))
               .div_((double)(n - 1))   // ∈ [0, 1]
               .reshape({1, 3, imgsz, imgsz});
  return x.to(dev);
}

torch::Tensor read_bin(const std::string& path,
                       const std::vector<int64_t>& shape) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float> buf((size_t)n);
  f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(n * sizeof(float)));
  if (!f) throw std::runtime_error("short read on " + path);
  auto t = torch::from_blob(buf.data(),
                            torch::IntArrayRef(shape),
                            torch::kFloat32).clone();
  return t;
}

struct ManifestEntry {
  int              idx;
  std::string      kind;
  std::vector<int64_t> shape;
};

std::vector<ManifestEntry> read_manifest(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<ManifestEntry> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    ManifestEntry e;
    std::string shape_csv;
    std::istringstream ss(line);
    ss >> e.idx >> e.kind >> shape_csv;
    std::stringstream sc(shape_csv);
    std::string tok;
    while (std::getline(sc, tok, ',')) e.shape.push_back(std::stoll(tok));
    out.push_back(std::move(e));
  }
  return out;
}

// Walks our Yolo11Detect's ModuleList layer-by-layer, capturing each
// layer's output tensor. Uses the same yaml topology as Yolo11DetectImpl.
struct V11Walker {
  // Per yolo11.cpp / yolo11_tasks.cpp v11 yaml.
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step>& yaml() {
    static const std::vector<Step> y = {
        {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},
        {{-1}, "C2PSA"},
        {{-1}, "Upsample"}, {{-1, 6},  "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Upsample"}, {{-1, 4},  "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Conv"},     {{-1, 13}, "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Conv"},     {{-1, 10}, "Concat"}, {{-1}, "C3k2"},
        {{16, 19, 22}, "Detect"},
    };
    return y;
  }

  static std::vector<torch::Tensor> capture(M::Yolo11Detect& mh,
                                             torch::Tensor x) {
    const auto& y = yaml();
    std::vector<torch::Tensor> outs(y.size());
    auto& mlist = mh->model;
    for (size_t i = 0; i < y.size(); ++i) {
      const auto& s = y[i];
      torch::Tensor in;
      if (s.kind == "Concat") {
        std::vector<torch::Tensor> parts;
        for (int f : s.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
        in = torch::cat(parts, 1);
        outs[i] = in;
      } else if (s.kind == "Detect") {
        std::vector<torch::Tensor> det_in;
        for (int f : s.from) det_in.push_back(outs[f]);
        auto* d = mlist[i]->as<M::DetectImpl>();
        // For Detect, capture the concat-form per-level + decoded view.
        // The Python hook returned a tuple/list — first element is the
        // decoded prediction. We mirror that by running decode.
        auto feats = d->forward_features(det_in);
        outs[i] = d->decode(feats);
      } else {
        int f = s.from[0];
        in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
        if      (s.kind == "Conv")     outs[i] = mlist[i]->as<M::ConvImpl>()->forward(in);
        else if (s.kind == "C3k2")     outs[i] = mlist[i]->as<M::C3k2Impl>()->forward(in);
        else if (s.kind == "SPPF")     outs[i] = mlist[i]->as<M::SPPFImpl>()->forward(in);
        else if (s.kind == "C2PSA")    outs[i] = mlist[i]->as<M::C2PSAImpl>()->forward(in);
        else if (s.kind == "Upsample") outs[i] = mlist[i]->as<torch::nn::UpsampleImpl>()->forward(in);
      }
    }
    return outs;
  }
};

struct V26Walker {
  struct Step { std::vector<int> from; std::string kind; };
  static const std::vector<Step>& yaml() {
    static const std::vector<Step> y = {
        {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"},
        {{-1}, "Conv"}, {{-1}, "C3k2"}, {{-1}, "SPPF"},
        {{-1}, "C2PSA"},
        {{-1}, "Upsample"}, {{-1, 6},  "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Upsample"}, {{-1, 4},  "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Conv"},     {{-1, 13}, "Concat"}, {{-1}, "C3k2"},
        {{-1}, "Conv"},     {{-1, 10}, "Concat"}, {{-1}, "C2PSAf"},
        {{16, 19, 22}, "Detect"},
    };
    return y;
  }

  static std::vector<torch::Tensor> capture(M::Yolo26Detect& mh,
                                             torch::Tensor x) {
    const auto& y = yaml();
    std::vector<torch::Tensor> outs(y.size());
    auto& mlist = mh->model;
    for (size_t i = 0; i < y.size(); ++i) {
      const auto& s = y[i];
      torch::Tensor in;
      if (s.kind == "Concat") {
        std::vector<torch::Tensor> parts;
        for (int f : s.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
        in = torch::cat(parts, 1);
        outs[i] = in;
      } else if (s.kind == "Detect") {
        std::vector<torch::Tensor> det_in;
        for (int f : s.from) det_in.push_back(outs[f]);
        auto* d = mlist[i]->as<M::Detect26Impl>();
        d->stride = mh->stride;
        auto feats = d->forward_features(det_in);
        outs[i] = d->decode(feats);
      } else {
        int f = s.from[0];
        in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
        if      (s.kind == "Conv")     outs[i] = mlist[i]->as<M::ConvImpl>()->forward(in);
        else if (s.kind == "C3k2")     outs[i] = mlist[i]->as<M::C3k2Impl>()->forward(in);
        else if (s.kind == "SPPF")     outs[i] = mlist[i]->as<M::SPPFImpl>()->forward(in);
        else if (s.kind == "C2PSA")    outs[i] = mlist[i]->as<M::C2PSAImpl>()->forward(in);
        else if (s.kind == "C2PSAf")   outs[i] = mlist[i]->as<M::C2PSAfImpl>()->forward(in);
        else if (s.kind == "Upsample") outs[i] = mlist[i]->as<torch::nn::UpsampleImpl>()->forward(in);
      }
    }
    return outs;
  }
};

void compare_outputs(const std::vector<torch::Tensor>& ours,
                     const std::vector<ManifestEntry>& manifest,
                     const std::string& dump_dir) {
  const double TOL = 1e-3;
  bool first_div_reported = false;
  std::printf("%-4s  %-10s  %-22s  %14s  %14s  %s\n",
              "idx", "kind", "shape", "max|Δ|", "mean|Δ|", "status");
  for (const auto& e : manifest) {
    char path[64];
    std::snprintf(path, sizeof(path), "%s/%03d.bin", dump_dir.c_str(), e.idx);
    auto py = read_bin(path, e.shape).to(torch::kCPU);
    auto cpp = ours[e.idx].detach().to(torch::kCPU).to(torch::kFloat32);
    // Skip layers whose output convention legitimately differs (e.g.
    // v26 Detect returns the e2e NMS-free [1, 300, 6] format from Python
    // but our decode returns [1, 4+nc, A] — same predictions, different
    // post-processing convention, can't compare elementwise).
    if (py.sizes() != cpp.sizes()) {
      std::ostringstream sa, sb;
      for (size_t k = 0; k < e.shape.size(); ++k) {
        if (k) sa << ',';
        sa << e.shape[k];
      }
      for (int k = 0; k < cpp.dim(); ++k) {
        if (k) sb << ',';
        sb << cpp.size(k);
      }
      std::printf("%-4d  %-10s  %-22s  %14s  %14s  shape-mismatch py=%s cpp=%s\n",
                  e.idx, e.kind.c_str(), sa.str().c_str(), "-", "-",
                  sa.str().c_str(), sb.str().c_str());
      continue;
    }
    auto diff = (py - cpp).abs();
    double max_d  = diff.max().item<double>();
    double mean_d = diff.mean().item<double>();
    bool ok = max_d < TOL;
    std::ostringstream ss;
    for (size_t k = 0; k < e.shape.size(); ++k) {
      if (k) ss << ',';
      ss << e.shape[k];
    }
    std::string status = ok ? "ok"
                            : (!first_div_reported ? "FIRST DIVERGENCE" : "diverged");
    if (!ok) first_div_reported = true;
    std::printf("%-4d  %-10s  %-22s  %14.6g  %14.6g  %s\n",
                e.idx, e.kind.c_str(), ss.str().c_str(),
                max_d, mean_d, status.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: parity_compare <version> <weights.pt> <dump_dir> [imgsz=640]\n";
    return 2;
  }
  std::string version  = argv[1];
  std::string weights  = argv[2];
  std::string dump_dir = argv[3];
  int imgsz = (argc > 4) ? std::stoi(argv[4]) : 640;

  auto manifest = read_manifest(dump_dir + "/manifest.txt");
  // Force CPU on both sides so the comparison isn't masked by cudnn vs.
  // our forward numerical drift — we want exact-bit parity at fp32.
  auto dev = torch::Device(torch::kCPU);
  auto x = fixed_input(imgsz, dev);

  // Use the same scale-inference path as the rest of the CLI (state-dict
  // probe + depth disambiguation for the v11/v26 m-vs-l ch=64 case).
  auto info = yolocpp::cli::infer_model_info(weights);
  if (version == "v11") {
    auto sd = yolocpp::serialization::load_state_dict(weights);
    auto scale = M::yolo11_scale_from_letter(info.scale);
    M::Yolo11Detect m(scale, 80);
    m->load_from_state_dict(sd.entries);
    m->to(dev); m->eval();
    torch::NoGradGuard ng;
    auto outs = V11Walker::capture(m, x);
    compare_outputs(outs, manifest, dump_dir);
  } else if (version == "v26") {
    auto sd = yolocpp::serialization::load_state_dict(weights);
    auto scale = M::yolo26_scale_from_letter(info.scale);
    M::Yolo26Detect m(scale, 80);
    m->load_from_state_dict(sd.entries);
    m->to(dev); m->eval();
    torch::NoGradGuard ng;
    auto outs = V26Walker::capture(m, x);
    compare_outputs(outs, manifest, dump_dir);
  } else {
    std::cerr << "unknown version: " << version << " (expected v11 or v26)\n";
    return 2;
  }
  return 0;
}
