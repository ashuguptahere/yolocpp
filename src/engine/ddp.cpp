// Distributed Data Parallel implementation using c10d (libtorch's bundled
// distributed comms library). Falls back to a no-op when WORLD_SIZE is unset.
//
// This file uses the c10d C++ API directly because libtorch doesn't expose
// `torch::nn::DistributedDataParallel` for C++ users — only Python. We
// implement the parts we need (broadcast at init + all-reduce after backward).

#include "yolocpp/engine/ddp.hpp"

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

namespace yolocpp::engine {

namespace {

struct GroupHolder {
  c10::intrusive_ptr<c10d::ProcessGroupNCCL> group;
  c10::intrusive_ptr<c10d::TCPStore>          store;
};

const char* getenv_or(const char* name, const char* dflt) {
  const char* v = std::getenv(name);
  return v ? v : dflt;
}

GroupHolder& impl_of(const DDPState& s) {
  if (!s.impl)
    throw std::runtime_error("DDP impl missing — init_ddp_from_env first");
  return *static_cast<GroupHolder*>(s.impl.get());
}

}  // anonymous namespace

DDPState init_ddp_from_env() {
  DDPState st;
  const char* world_env = std::getenv("WORLD_SIZE");
  if (!world_env || std::stoi(world_env) <= 1) {
    return st;  // single-process; .active = false
  }
  st.world_size  = std::stoi(world_env);
  st.rank        = std::stoi(getenv_or("RANK", "0"));
  st.local_rank  = std::stoi(getenv_or("LOCAL_RANK", "0"));
  st.master_addr = getenv_or("MASTER_ADDR", "127.0.0.1");
  st.master_port = std::stoi(getenv_or("MASTER_PORT", "29500"));

  if (st.local_rank >= 0) c10::cuda::set_device(st.local_rank);

  // Build a TCPStore (rank 0 hosts), wrap in a PrefixStore so the
  // process group has its own namespace.
  c10d::TCPStoreOptions opts;
  opts.port            = st.master_port;
  opts.numWorkers      = st.world_size;
  opts.isServer        = (st.rank == 0);
  opts.waitWorkers     = true;
  opts.timeout         = std::chrono::milliseconds(60'000);

  auto store_raw = c10::make_intrusive<c10d::TCPStore>(st.master_addr, opts);
  auto prefix    = c10::make_intrusive<c10d::PrefixStore>("yolocpp/", store_raw);

  c10d::ProcessGroupNCCL::Options pg_opts;
  pg_opts.timeout = std::chrono::milliseconds(60'000);
  auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(
      prefix, st.rank, st.world_size,
      c10::make_intrusive<c10d::ProcessGroupNCCL::Options>(pg_opts));

  auto holder = std::make_shared<GroupHolder>();
  holder->group = pg;
  holder->store = store_raw;
  st.impl   = holder;
  st.active = true;

  if (st.rank == 0) {
    std::cout << "[ddp] initialised: world_size=" << st.world_size
              << " master=" << st.master_addr << ":" << st.master_port << "\n";
  }
  return st;
}

void finalize_ddp(DDPState& state) {
  if (!state.active) return;
  // The ProcessGroup destructor runs comms-shutdown when impl is reset.
  state.impl.reset();
  state.active = false;
}

void barrier(const DDPState& state) {
  if (!state.active) return;
  auto& h = impl_of(state);
  c10d::BarrierOptions opts;
  h.group->barrier(opts)->wait();
}

void broadcast_module(const DDPState& state, torch::nn::Module& model) {
  if (!state.active) return;
  auto& h = impl_of(state);
  std::vector<at::Tensor> tensors;
  for (auto& kv : model.named_parameters()) tensors.push_back(kv.value());
  for (auto& kv : model.named_buffers())    tensors.push_back(kv.value());
  c10d::BroadcastOptions opts;
  opts.rootRank = 0;
  h.group->broadcast(tensors, opts)->wait();
}

void all_reduce_grads(const DDPState& state, torch::nn::Module& model) {
  if (!state.active) return;
  auto& h = impl_of(state);
  std::vector<at::Tensor> grads;
  grads.reserve(64);
  for (auto& kv : model.named_parameters()) {
    auto& p = kv.value();
    if (!p.grad().defined()) continue;
    grads.push_back(p.grad());
  }
  if (grads.empty()) return;
  c10d::AllreduceOptions opts;
  opts.reduceOp = c10d::ReduceOp::SUM;
  h.group->allreduce(grads, opts)->wait();
  // Average — divide by world_size in-place.
  double inv = 1.0 / state.world_size;
  for (auto& g : grads) g.mul_(inv);
}

std::vector<int64_t> distributed_indices(const DDPState& state, int64_t n,
                                         int epoch) {
  // Deterministic shuffle keyed by epoch, then take the rank-th slice.
  std::vector<int64_t> all(n);
  for (int64_t i = 0; i < n; ++i) all[i] = i;
  std::mt19937 rng(0x9E3779B9u ^ (uint32_t)epoch);
  std::shuffle(all.begin(), all.end(), rng);
  if (!state.active) return all;
  // Pad so every rank gets the same count (drop_last=False analogue).
  int64_t per_rank = (n + state.world_size - 1) / state.world_size;
  while ((int64_t)all.size() < per_rank * state.world_size)
    all.push_back(all[(int)all.size() % n]);
  std::vector<int64_t> mine;
  mine.reserve(per_rank);
  for (int64_t i = state.rank; i < (int64_t)all.size(); i += state.world_size)
    mine.push_back(all[i]);
  return mine;
}

}  // namespace yolocpp::engine
