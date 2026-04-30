// Dev-only utility: dump every state_dict entry of a .pt as
//   <key>  <dtype>  <shape>
// plus the total parameter count.

#include <iostream>
#include <numeric>
#include <string>

#include "yolocpp/serialization/pt_loader.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: inspect_pt <file.pt>\n"; return 2; }
  auto sd = yolocpp::serialization::load_state_dict(argv[1]);
  long long total = 0;
  for (const auto& [k, t] : sd.entries) {
    long long n = 1;
    for (auto d : t.sizes()) n *= d;
    total += n;
    std::cout << k << "  " << t.dtype() << "  " << t.sizes() << "\n";
  }
  std::cout << "----- " << sd.entries.size() << " tensors, "
            << total << " elements\n";
  return 0;
}
