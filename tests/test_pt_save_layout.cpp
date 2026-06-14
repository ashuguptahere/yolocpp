// Regression test for the channels_last (NHWC) save-layout bug.
//
// `save_state_dict` writes raw storage bytes from a `.contiguous()` copy of
// each tensor, so the sizes/strides recorded in `data.pkl` MUST describe that
// same row-major-contiguous layout. A 4D conv weight in channels_last memory
// format (which the trainer puts every conv param into on CUDA) previously had
// its NHWC strides written over NCHW-contiguous bytes — corrupting the tensor
// for any strides-respecting reader (PyTorch / upstream interop), even though
// our own strides-agnostic loader masked it.
//
// We assert layout-normalization the robust, version-independent way: saving a
// channels_last tensor and its `.contiguous()` twin must produce a byte-
// identical `data.pkl` record (same sizes + strides), and the round-trip must
// preserve values. Before the fix the two records differ (NHWC vs NCHW
// strides); after, they are identical.

#include <torch/torch.h>

#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/file_adapter.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace {

// Read the raw `data.pkl` record (the pickle holding sizes + strides) from a
// saved .pt zip.
std::vector<uint8_t> read_data_pkl(const std::string& path) {
  caffe2::serialize::PyTorchStreamReader reader(path);
  auto [ptr, size] = reader.getRecord("data.pkl");
  std::vector<uint8_t> out(size);
  std::memcpy(out.data(), ptr.get(), size);
  return out;
}

int g_failures = 0;
void check(bool ok, const std::string& msg) {
  std::cout << (ok ? "[ok]   " : "[FAIL] ") << msg << "\n";
  if (!ok) ++g_failures;
}

}  // namespace

int main() {
  using namespace yolocpp::serialization;

  // A 4D ramp tensor whose channels_last and contiguous strides genuinely
  // differ (C>1, H*W>1).
  auto w = torch::arange(2 * 3 * 4 * 5, torch::kFloat).reshape({2, 3, 4, 5});
  auto w_cl = w.contiguous(at::MemoryFormat::ChannelsLast);

  check(!w_cl.is_contiguous(at::MemoryFormat::Contiguous),
        "channels_last input is non-contiguous in NCHW order (precondition)");

  const std::string p_cl = "/tmp/yolocpp_layout_cl.pt";
  const std::string p_co = "/tmp/yolocpp_layout_co.pt";
  save_state_dict(p_cl, {{"w", w_cl}});
  save_state_dict(p_co, {{"w", w.contiguous()}});

  // Core assertion: the layout metadata is normalized, so the two pickles are
  // byte-identical. This fails if the emitter ever records the source tensor's
  // (NHWC) strides instead of the written (NCHW) storage layout.
  auto pkl_cl = read_data_pkl(p_cl);
  auto pkl_co = read_data_pkl(p_co);
  check(pkl_cl == pkl_co,
        "data.pkl of channels_last save == contiguous save (strides match bytes)");

  // And the round-trip preserves the logical values.
  auto sd  = load_state_dict(p_cl);
  auto got = sd.entries.at(0).second.reshape(-1).to(torch::kFloat);
  check(torch::allclose(got, w.reshape(-1).to(torch::kFloat)),
        "channels_last save round-trips to the correct NCHW values");

  std::cout << (g_failures == 0 ? "ALL PASS\n" : "FAILURES\n");
  return g_failures == 0 ? 0 : 1;
}
