// Per-layer divergence localizer for v13l. Forwards through Yolo13Detect
// while sampling each YAML layer's output and compares against
// dumps/yolo13l/<idx>.bin. Prints first divergent layer.

#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

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
  return torch::from_blob(buf.data(), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

static std::vector<int64_t> read_shape_from_manifest(const std::string& path,
                                                       int idx) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string line;
  while (std::getline(f, line)) {
    int li;
    char tab1, tab2;
    std::string kind;
    std::istringstream iss(line);
    iss >> li;
    if (li != idx) continue;
    iss >> kind;
    std::string shape_str;
    iss >> shape_str;
    std::vector<int64_t> shape;
    int64_t v = 0;
    for (char c : shape_str) {
      if (c == ',') { shape.push_back(v); v = 0; }
      else          { v = v * 10 + (c - '0'); }
    }
    shape.push_back(v);
    return shape;
  }
  throw std::runtime_error("manifest missing idx " + std::to_string(idx));
}

int main() {
  const std::string pt_path  = "data/yolo13l.pt";
  const std::string dump_dir = "/tmp/yolocpp_parity/dumps/yolo13l";
  if (!fs::exists(dump_dir + "/032.bin")) {
    std::printf("[skip] no v13l dumps\n");
    return 0;
  }

  yolocpp::models::Yolo13Detect m(yolocpp::models::kYolo13l, /*nc=*/80);
  auto sd = yolocpp::serialization::load_state_dict(pt_path);
  m->load_from_state_dict(sd.entries);
  m->eval();

  int imgsz = 640;
  int64_t N = 3LL * imgsz * imgsz;
  auto x = torch::arange(N, torch::TensorOptions().dtype(torch::kFloat32))
               .div_(static_cast<double>(N - 1))
               .view({1, 3, imgsz, imgsz})
               .contiguous();

  // Hook all 33 child modules and capture their output tensors.
  std::vector<torch::Tensor> caps(33);
  std::vector<torch::jit::Module> dummy;  // unused
  // libtorch C++ API doesn't have register_forward_hook on nn::Module the
  // same way as Python. Instead we cherry-pick the per-layer outputs by
  // calling forward_train (which returns just the head feature levels) —
  // not enough. We need access to outs.
  //
  // Simpler: re-run the layer schedule manually here, mirroring
  // forward_train, and capture each output.
  static const struct LStep { std::vector<int> from; std::string kind; }
      yaml[] = {
          {{-1}, "Conv"}, {{-1}, "Conv"}, {{-1}, "DSC3k2"},
          {{-1}, "Conv"}, {{-1}, "DSC3k2"},
          {{-1}, "DSConv"}, {{-1}, "A2C2f"},
          {{-1}, "DSConv"}, {{-1}, "A2C2f"},
          {{4,6,8}, "HyperACE"},
          {{-1}, "Up"}, {{9}, "DownsampleConv"},
          {{6,9}, "FullPADTunnel"}, {{4,10}, "FullPADTunnel"},
          {{8,11}, "FullPADTunnel"},
          {{-1}, "Up"}, {{-1,12}, "Cat"}, {{-1}, "DSC3k2"},
          {{-1,9}, "FullPADTunnel"},
          {{17}, "Up"}, {{-1,13}, "Cat"}, {{-1}, "DSC3k2"},
          {{10}, "Conv"}, {{21,22}, "FullPADTunnel"},
          {{-1}, "Conv"}, {{-1,18}, "Cat"}, {{-1}, "DSC3k2"},
          {{-1,9}, "FullPADTunnel"},
          {{26}, "Conv"}, {{-1,14}, "Cat"}, {{-1}, "DSC3k2"},
          {{-1,11}, "FullPADTunnel"},
          {{23,27,31}, "Detect"},
      };

  std::vector<torch::Tensor> outs(33);
  torch::Tensor prev = x;
  torch::NoGradGuard ng;

  auto resolve = [&](int f) {
    return f == -1 ? prev : outs[f];
  };
  using namespace yolocpp::models;
  for (size_t i = 0; i < 33; ++i) {
    auto mod = m->model[i];
    const auto& s = yaml[i];
    torch::Tensor y;
    if (s.kind == "Conv") {
      y = mod->as<ConvImpl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "DSConv") {
      y = mod->as<DSConvImpl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "DSC3k2") {
      y = mod->as<DSC3k2Impl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "A2C2f") {
      y = mod->as<V13A2C2fImpl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "Up") {
      y = mod->as<torch::nn::UpsampleImpl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "DownsampleConv") {
      y = mod->as<DownsampleConvImpl>()->forward(resolve(s.from[0]));
    } else if (s.kind == "HyperACE") {
      std::vector<torch::Tensor> ins;
      for (int f : s.from) ins.push_back(outs[f]);
      y = mod->as<HyperACEImpl>()->forward(ins);
    } else if (s.kind == "FullPADTunnel") {
      auto a = resolve(s.from[0]);
      auto b = resolve(s.from[1]);
      y = mod->as<FullPADTunnelImpl>()->forward(a, b);
    } else if (s.kind == "Cat") {
      std::vector<torch::Tensor> ins;
      for (int f : s.from) ins.push_back(resolve(f));
      y = torch::cat(ins, /*dim=*/1);
    } else if (s.kind == "Detect") {
      auto* det = mod->as<DetectImpl>();
      std::vector<torch::Tensor> feats;
      for (int f : s.from) feats.push_back(outs[f]);
      auto raw = det->forward_features(feats);
      for (size_t k = 0; k < raw.size(); ++k) {
        std::printf("  Detect raw[%zu] shape=%s min=%.4e max=%.4e\n", k,
                    std::string(c10::str(raw[k].sizes())).c_str(),
                    raw[k].min().item<double>(),
                    raw[k].max().item<double>());
      }
      auto decoded = det->decode(raw);
      std::printf("  Detect decoded shape=%s\n",
                  std::string(c10::str(decoded.sizes())).c_str());
      auto cls = decoded.slice(/*dim=*/1, /*start=*/4, /*end=*/84);
      std::printf("  cls min=%.4e max=%.4e mean=%.4e\n",
                  cls.min().item<double>(),
                  cls.max().item<double>(),
                  cls.mean().item<double>());
      break;
    }
    outs[i] = y;
    prev = y;

    // Compare against Python dump.
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/%03zu.bin", dump_dir.c_str(), i);
    if (!fs::exists(buf)) continue;
    auto shape = read_shape_from_manifest(dump_dir + "/manifest.txt",
                                            (int)i);
    if (shape != std::vector<int64_t>(y.sizes().begin(), y.sizes().end())) {
      std::printf("[L%02zu shape mismatch] ours=%s ref=%s kind=%s\n", i,
                  std::string(c10::str(y.sizes())).c_str(),
                  std::string(c10::str(at::IntArrayRef(shape))).c_str(),
                  s.kind.c_str());
      return 1;
    }
    auto ref = read_bin(buf, shape);
    auto diff = (y - ref).abs();
    double m_abs = diff.max().item<double>();
    double mean_abs = diff.mean().item<double>();
    std::printf("L%02zu %-15s max|Δ|=%.3e mean|Δ|=%.3e shape=%s\n",
                i, s.kind.c_str(), m_abs, mean_abs,
                std::string(c10::str(y.sizes())).c_str());
    if (m_abs > 1e-2) {
      std::printf("[diverged at L%02zu]\n", i);
      return 0;
    }
  }
  return 0;
}
