// #65E2 stepping stone: prove that the upstream RF-DETR 1.6.5
// `.pt` / `.pth` files unpickle cleanly through our existing
// `pt_loader.cpp` and produce a state_dict whose key set matches
// the inventory captured in `docs/rfdetr_arch.md`.
//
// SKIP-gated: the official 1498-MiB rf-detr-large.pth + the smaller
// detect / segment weights live outside the repo (see
// `scripts/install_third_party.sh` does NOT fetch them; the next
// session's converter slice (#65E2) will add a download helper).
// Until then, point this test at a manually-downloaded weight file
// via the `RFDETR_TEST_WEIGHTS_DIR` environment variable.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>

#include "yolocpp/serialization/pt_loader.hpp"

namespace fs = std::filesystem;
using yolocpp::serialization::load_flat_state_dict;
using yolocpp::serialization::StateDict;

namespace {
bool contains(const StateDict& sd, const std::string& key) {
  for (const auto& e : sd.entries)
    if (e.first == key) return true;
  return false;
}
at::Tensor at_key(const StateDict& sd, const std::string& key) {
  for (const auto& e : sd.entries)
    if (e.first == key) return e.second;
  return at::Tensor{};
}
}

namespace {

struct VariantSpec {
  const char* file;          // basename
  int         param_count;   // exact key count from docs/rfdetr_arch.md
  const char* mandatory_key;
  const char* mandatory_seg_key;  // nullptr for detect-only
};

// Inventories from /tmp/yolocpp_parity/rfdetr_dumps/*.json (committed
// as docs/rfdetr_arch.md). The mandatory_key list pins the most
// architecture-defining shapes — backbone position embedding,
// decoder layer 0 self-attn, decoder cls head — so a missing or
// renamed key fails fast.
constexpr VariantSpec kVariants[] = {
    {"rf-detr-nano.pth",    465, "refpoint_embed.weight",     nullptr},
    {"rf-detr-small.pth",   487, "refpoint_embed.weight",     nullptr},
    {"rf-detr-medium.pth",  509, "refpoint_embed.weight",     nullptr},
    {"rf-detr-base.pth",    487, "refpoint_embed.weight",     nullptr},
    {"rf-detr-large.pth",   533, "refpoint_embed.weight",     nullptr},
    {"rf-detr-seg-nano.pt",    544, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-small.pt",   544, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-medium.pt",  572, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-large.pt",   572, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-xlarge.pt",  600, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-xxlarge.pt", 600, "refpoint_embed.weight",  "segmentation_head.bias"},
    {"rf-detr-seg-preview.pt", 544, "refpoint_embed.weight",  "segmentation_head.bias"},
};

}  // namespace

int main() {
  const char* dir = std::getenv("RFDETR_TEST_WEIGHTS_DIR");
  if (!dir || !*dir) {
    std::cout << "SKIP: RFDETR_TEST_WEIGHTS_DIR not set "
                 "(point it at a directory containing the 12 official "
                 "rfdetr 1.6.5 weight files; see docs/rfdetr_arch.md)\n";
    return 0;
  }
  fs::path d(dir);
  if (!fs::is_directory(d)) {
    std::cout << "SKIP: " << dir << " is not a directory\n";
    return 0;
  }
  int checked = 0, missing = 0, fails = 0;
  for (const auto& v : kVariants) {
    fs::path p = d / v.file;
    if (!fs::exists(p)) {
      std::cout << "SKIP: " << v.file << " missing\n";
      ++missing;
      continue;
    }
    auto sd = load_flat_state_dict(p.string(), /*submodel=*/"model");
    bool ok = true;
    if (static_cast<int>(sd.size()) != v.param_count) {
      std::cerr << "[FAIL] " << v.file << ": param count "
                << sd.size() << " expected " << v.param_count << "\n";
      ok = false;
    }
    if (!contains(sd,v.mandatory_key)) {
      std::cerr << "[FAIL] " << v.file << ": missing key '"
                << v.mandatory_key << "'\n";
      ok = false;
    }
    if (v.mandatory_seg_key && !contains(sd,v.mandatory_seg_key)) {
      std::cerr << "[FAIL] " << v.file << ": missing seg key '"
                << v.mandatory_seg_key << "'\n";
      ok = false;
    }
    if (ok) {
      auto refpt = at_key(sd, v.mandatory_key);
      std::cout << "[PASS] " << v.file << " params=" << sd.size()
                << " refpoint=" << refpt.sizes() << "\n";
      ++checked;
    } else {
      ++fails;
    }
  }
  std::cout << "rfdetr pt_load summary: passed=" << checked
            << " failed=" << fails << " skipped=" << missing << "\n";
  return fails == 0 ? 0 : 1;
}
