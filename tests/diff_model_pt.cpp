// Read-only diagnostic: diff a constructed model's parameter names against a
// checkpoint's keys, to find which model params won't load (→ stay at random
// init). Usage: diff_model_pt <v9|v10> <scale-letter> <file.pt>
#include <iostream>
#include <set>
#include <string>

#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: diff_model_pt <v9|v10> <scale> <file.pt>\n"; return 2; }
  std::string ver = argv[1], scale = argv[2], pt = argv[3];

  std::set<std::string> model_names;
  auto collect = [&](auto& holder) {
    for (const auto& p : holder->named_parameters(true)) model_names.insert(p.key());
    for (const auto& b : holder->named_buffers(true))    model_names.insert(b.key());
  };
  if (ver == "v9") {
    yolocpp::models::Yolo9 m(yolocpp::models::yolo9_scale_from_letter(scale), 80);
    collect(m);
  } else if (ver == "v10") {
    yolocpp::models::Yolo10 m(yolocpp::models::yolo10_scale_from_letter(scale), 80);
    collect(m);
  } else { std::cerr << "unknown version\n"; return 2; }

  auto sd = yolocpp::serialization::load_state_dict(pt);
  std::set<std::string> pt_names;
  for (const auto& [k, t] : sd.entries) pt_names.insert(k);

  int unloaded = 0, ignored = 0;
  std::cout << "=== MODEL params NOT in .pt (stay random-init → break inference) ===\n";
  for (const auto& n : model_names)
    if (!pt_names.count(n)) { std::cout << "  " << n << "\n"; ++unloaded; }
  std::cout << "=== .pt keys NOT in MODEL (ignored on load) ===\n";
  for (const auto& n : pt_names)
    if (!model_names.count(n)) { std::cout << "  " << n << "\n"; ++ignored; }
  std::cout << "----- model params=" << model_names.size()
            << "  pt keys=" << pt_names.size()
            << "  unloaded(model-only)=" << unloaded
            << "  ignored(pt-only)=" << ignored << "\n";
  return 0;
}
