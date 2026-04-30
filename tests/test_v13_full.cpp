// End-to-end v13n parity: load the .pt, forward our Yolo13Detect,
// compare against dump.py's layer-32 (Detect) output.
//
// Skips cleanly if dumps are absent.

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

static int run_scale(const std::string& letter,
                      const yolocpp::models::Yolo13Scale& sc) {
  std::string pt_path  = "data/yolo13" + letter + ".pt";
  std::string dump_dir = "/tmp/yolocpp_parity/dumps/yolo13" + letter;
  if (!fs::exists(pt_path) || !fs::exists(dump_dir + "/032.bin")) {
    std::printf("[skip] yolo13%s missing weights or dump\n", letter.c_str());
    return 0;
  }
  yolocpp::models::Yolo13Detect m(sc, /*nc=*/80);
  auto sd = yolocpp::serialization::load_state_dict(pt_path);
  int loaded = m->load_from_state_dict(sd.entries);
  m->eval();

  int imgsz = 640;
  int64_t N = 3LL * imgsz * imgsz;
  auto x = torch::arange(N, torch::TensorOptions().dtype(torch::kFloat32))
               .div_(static_cast<double>(N - 1))
               .view({1, 3, imgsz, imgsz})
               .contiguous();

  torch::NoGradGuard ng;
  auto out_ours = m->forward_eval(x);
  auto ref = read_bin(dump_dir + "/032.bin", {1, 84, 8400});
  auto cls_ours = out_ours.slice(/*dim=*/1, /*start=*/4, /*end=*/84);
  auto cls_ref  = ref.slice(/*dim=*/1, /*start=*/4, /*end=*/84);
  auto diff_cls = (cls_ours - cls_ref).abs();
  double max_cls  = diff_cls.max().item<double>();
  double mean_cls = diff_cls.mean().item<double>();
  std::printf("[v13/full] yolo13%s loaded=%d cls max|Δ|=%.3e mean|Δ|=%.3e\n",
              letter.c_str(), loaded, max_cls, mean_cls);
  if (max_cls > 1e-3) {
    std::printf("[fail] yolo13%s cls-channels diverge\n", letter.c_str());
    return 1;
  }
  return 0;
}

int main() {
  int rc = 0;
  rc |= run_scale("n", yolocpp::models::kYolo13n);
  rc |= run_scale("s", yolocpp::models::kYolo13s);
  rc |= run_scale("l", yolocpp::models::kYolo13l);
  rc |= run_scale("x", yolocpp::models::kYolo13x);
  if (rc == 0) std::printf("[pass]\n");
  return rc;
}
