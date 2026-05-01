// Dev tool — dump Yolo10's C++ forward_eval output for a fixed
// arange-based input, for ONNX parity comparison vs onnxruntime.
//
// Usage: dump_v10_forward <yolo10_pt> <scale_letter> <out_bin>
// where scale_letter ∈ {n,s,m,b,l,x}. Writes raw fp32 bytes in
// [1, 84, 8400] layout (channel-major).

#include <torch/torch.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: dump_v10_forward <yolo10_pt> <scale> <out_bin>\n";
    return 2;
  }
  using namespace yolocpp;
  std::string pt = argv[1];
  std::string scale_letter = argv[2];
  std::string out_path = argv[3];

  auto scale = models::yolo10_scale_from_letter(scale_letter);
  models::Yolo10 m(scale, 80);
  auto sd = serialization::load_state_dict(pt);
  m->load_from_state_dict(sd.entries);
  m->eval();

  int N = 3 * 640 * 640;
  auto x = torch::arange(N, torch::kFloat32).div(float(N - 1)).reshape({1, 3, 640, 640});
  torch::Tensor pred;
  {
    torch::NoGradGuard ng;
    pred = m->forward_eval(x);
  }
  pred = pred.contiguous();
  std::cout << "shape=[" << pred.size(0) << "," << pred.size(1)
            << "," << pred.size(2) << "] "
            << "min=" << pred.min().item<float>() << " max="
            << pred.max().item<float>() << "\n";

  std::ofstream f(out_path, std::ios::binary);
  if (!f) { std::cerr << "cannot write " << out_path << "\n"; return 1; }
  f.write(reinterpret_cast<const char*>(pred.data_ptr<float>()),
          pred.numel() * sizeof(float));
  std::cout << "wrote " << pred.numel() * sizeof(float) << " bytes to "
            << out_path << "\n";
  return 0;
}
