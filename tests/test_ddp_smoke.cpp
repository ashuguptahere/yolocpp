// DDP infrastructure smoke test.
//
// What this verifies (single-GPU box):
//   • init_ddp_from_env() returns inactive when WORLD_SIZE is unset
//   • all_reduce_grads / broadcast_module / barrier are no-ops in that mode
//   • distributed_indices(N=20, single-rank) returns all 20 shuffled
//   • finalize_ddp() is safe to call on an inactive state
//
// What CANNOT be verified here (needs 2+ GPUs / hosts):
//   • NCCL collective correctness across ranks
//   • Gradient agreement between 1-GPU and 2-GPU training
//
// For full multi-GPU validation:
//   scripts/launch_ddp.sh 2 task=detect mode=train data=coco8 model=yolov8n.pt

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <set>

#include "yolocpp/engine/ddp.hpp"
#include "yolocpp/models/yolov8.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  // Make sure no leaked DDP env vars affect the test.
  ::unsetenv("WORLD_SIZE");
  ::unsetenv("RANK");
  ::unsetenv("LOCAL_RANK");

  auto st = yolocpp::engine::init_ddp_from_env();
  EXPECT(!st.active,                    "default: DDP inactive");
  EXPECT(st.world_size == 1,            "default world_size == 1");
  EXPECT(st.rank == 0,                  "default rank == 0");
  EXPECT(yolocpp::engine::is_rank0(st), "is_rank0 true at single-process");

  // No-op helpers.
  yolocpp::models::YoloV8Detect m(yolocpp::models::kYoloV8n, /*nc=*/80);
  yolocpp::engine::broadcast_module(st, *m);
  yolocpp::engine::all_reduce_grads(st, *m);
  yolocpp::engine::barrier(st);

  // distributed_indices at single-rank returns all N indices.
  auto idx = yolocpp::engine::distributed_indices(st, 20, /*epoch=*/3);
  EXPECT((int)idx.size() == 20,         "single-rank: all 20 indices");
  std::set<int64_t> u(idx.begin(), idx.end());
  EXPECT(u.size() == 20,                "indices are unique");

  yolocpp::engine::finalize_ddp(st);
  EXPECT(!st.active,                    "post-finalize: inactive");

  std::cout << "=== ddp smoke test PASS ===\n";
  return 0;
}
