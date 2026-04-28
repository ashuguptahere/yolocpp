#pragma once
//
// Distributed Data Parallel (DDP) primitives — multi-GPU training via NCCL.
//
// The library is a small wrapper around c10d's ProcessGroupNCCL + TCPStore.
// It implements the bare minimum needed for synchronous data-parallel training:
//
//   • init_from_env()            → rendezvous via env vars (Ultralytics /
//                                   torchrun convention: RANK, WORLD_SIZE,
//                                   LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
//   • broadcast_module()         → copy rank-0 weights to all ranks at start
//   • all_reduce_grads(model)    → averages gradients across ranks after backward
//   • barrier(), is_rank0()      → small helpers
//   • DistributedIndices         → per-rank slice of dataset indices
//
// The launcher (scripts/launch_ddp.sh) sets the env vars and spawns one
// process per GPU. Within a single process, only the local GPU is used.
//
// At world_size=1 the entire path is a no-op (init_from_env returns false
// when RANK is unset), so the rest of the trainer compiles + runs unchanged.
//
// VERIFICATION STATUS:
//   • Compiles + tested at world_size=1 on a single 5090.
//   • For 2+ GPU validation, run on a multi-GPU machine with:
//       scripts/launch_ddp.sh 2 task=detect mode=train data=coco8 model=yolov8n.pt
//   • Bit-exact gradient agreement with single-GPU after `world_size` steps
//     hasn't been validated here (no second GPU); the all-reduce code path
//     follows the standard PyTorch DDP pattern.
//

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

namespace yolocpp::engine {

struct DDPState {
  bool        active     = false;
  int         rank       = 0;
  int         world_size = 1;
  int         local_rank = 0;
  std::string master_addr = "127.0.0.1";
  int         master_port = 29500;
  // Held opaquely; the implementation owns the c10d ProcessGroup.
  std::shared_ptr<void> impl;
};

// Initialise DDP from the standard env vars. Returns:
//   .active = true   if WORLD_SIZE > 1 and we successfully joined the group
//   .active = false  if running single-process (no env vars set)
// Throws on env-var inconsistency or NCCL init failure.
DDPState init_ddp_from_env();

// Tear down the process group cleanly. Safe to call when state.active == false.
void finalize_ddp(DDPState& state);

// Average parameter gradients across all ranks (in-place).
// No-op when state.active == false.
void all_reduce_grads(const DDPState& state, torch::nn::Module& model);

// Broadcast model parameters + buffers from rank 0 to all ranks (in-place).
// Used at the start of training so every rank has the same initial weights.
void broadcast_module(const DDPState& state, torch::nn::Module& model);

// Synchronisation barrier. No-op single-process.
void barrier(const DDPState& state);

// Return true on rank 0 (or single-process).
inline bool is_rank0(const DDPState& s) { return !s.active || s.rank == 0; }

// Per-rank slice of [0, n) for a given epoch:
// returns a deterministic shuffle of indices owned by this rank.
std::vector<int64_t> distributed_indices(const DDPState& state,
                                         int64_t n, int epoch);

}  // namespace yolocpp::engine
