// One-off: invoke a version's upstream→deploy reparam converter.
// Usage: run_convert <v6|v7|v9|v10> <src.pt> <out.pt> [nc]
#include <iostream>
#include <string>

#include "yolocpp/serialization/yolov6_weights.hpp"
#include "yolocpp/serialization/yolov7_weights.hpp"
#include "yolocpp/serialization/yolov9_weights.hpp"
#include "yolocpp/serialization/yolov10_weights.hpp"

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: run_convert <v6|v7|v9|v10> <src.pt> <out.pt> [nc]\n"; return 2; }
  std::string ver = argv[1], src = argv[2], out = argv[3];
  int nc = argc > 4 ? std::stoi(argv[4]) : 80;
  namespace s = yolocpp::serialization;
  int n = 0;
  if      (ver == "v6")  n = s::convert_yolov6_pt(src, out, nc);
  else if (ver == "v7")  n = s::convert_yolov7_pt(src, out, nc);
  else if (ver == "v9")  n = s::convert_yolov9_pt(src, out, nc);
  else if (ver == "v10") n = s::convert_yolov10_pt(src, out, nc);
  else { std::cerr << "unknown version\n"; return 2; }
  std::cout << "converted " << ver << ": wrote " << n << " tensors to " << out << "\n";
  return 0;
}
