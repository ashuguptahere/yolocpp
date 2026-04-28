#!/usr/bin/env bash
# Spawn one yolocpp process per GPU for synchronous DDP training.
#
# Usage:
#   scripts/launch_ddp.sh <num_gpus> <yolocpp args...>
#
# Examples:
#   scripts/launch_ddp.sh 2 task=detect mode=train data=coco model=yolo8n.pt epochs=50
#   scripts/launch_ddp.sh 4 task=detect mode=train data=coco model=yolo8x.pt batch=64
#
# The launcher sets WORLD_SIZE / RANK / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
# in each child process. The trainer's init_ddp_from_env() picks them up and
# joins the process group via NCCL + TCPStore.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <num_gpus> <yolocpp args...>" >&2
  exit 1
fi

NGPUS="$1"; shift
ARGS=("$@")

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/yolocpp"
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — did you run cmake --build build?" >&2
  exit 1
fi

# Pick a free port for the rendezvous server.
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

PIDS=()
trap 'kill ${PIDS[@]} 2>/dev/null || true' EXIT

LOG_DIR="${LOG_DIR:-runs/ddp_logs}"
mkdir -p "$LOG_DIR"

for ((rank = 0; rank < NGPUS; rank++)); do
  WORLD_SIZE="$NGPUS" \
  RANK="$rank" \
  LOCAL_RANK="$rank" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  CUDA_VISIBLE_DEVICES="$rank" \
  "$BIN" "${ARGS[@]}" device=cuda:0 \
      > "$LOG_DIR/rank${rank}.log" 2>&1 &
  PIDS+=("$!")
  echo "[launch_ddp] rank=$rank pid=${PIDS[-1]} log=$LOG_DIR/rank${rank}.log"
done

# Stream rank 0's log so the user has live progress.
tail -f "$LOG_DIR/rank0.log" &
TAIL_PID=$!

# Wait on all worker processes; bail (and kill the tail) when any exits.
EXIT=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then EXIT=$?; fi
done
kill "$TAIL_PID" 2>/dev/null || true
exit "$EXIT"
