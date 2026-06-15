#!/usr/bin/env bash
#
# train_matrix.sh — #60A train harness.
#
# Drives `yolocpp --mode train` across a (version × scale × task) matrix defined
# in a TSV manifest. Each cell trains to its own `runs/matrix/<cell>/` dir,
# resumable at cell granularity (a cell whose best.pt exists is skipped), with a
# per-cell log and a roll-up results CSV. Built for the #60 "retrain every
# (version × scale × task) on COCO and publish weights" effort: the harness is
# the engine, scripts/train_matrix.tsv is the work-list.
#
# This script does NOT source any environment — run it after your usual build
# env is active (so `./build/yolocpp` and its libs resolve), e.g.:
#   source /tmp/yolocpp_env.sh && bash scripts/train_matrix.sh --smoke
#
# Usage:
#   bash scripts/train_matrix.sh [options]
# Options:
#   --manifest FILE   work-list TSV (default: scripts/train_matrix.tsv)
#   --out DIR         output root   (default: runs/matrix)
#   --device DEV      --device value passed to training (default: cuda)
#   --smoke           tiny verification run: 2 epochs, batch 4, coco8* datasets
#   --epochs N        override every cell's epoch count
#   --filter REGEX    only run cells whose name matches REGEX
#   --export FMT      --export-after-train FMT (e.g. onnx, trt, onnx,trt)
#   --force           retrain even if best.pt exists
#   --dry-run         print the commands without running them
#   --bin PATH        yolocpp binary (default: ./build/yolocpp)
#
# Full-matrix datasets are taken from env vars (override the placeholders):
#   YOLOCPP_DATA_DETECT  YOLOCPP_DATA_SEGMENT  YOLOCPP_DATA_POSE
#   YOLOCPP_DATA_OBB     YOLOCPP_DATA_CLASSIFY
# In --smoke mode these are forced to the data/coco8* (+ data/) smoke sets.

set -uo pipefail

MANIFEST="scripts/train_matrix.tsv"
OUT="runs/matrix"
DEVICE="cuda"
SMOKE=0
DRYRUN=0
FORCE=0
FILTER=""
EPOCHS_OVERRIDE=""
EXPORT=""
BIN="./build/yolocpp"

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    --device)   DEVICE="$2"; shift 2 ;;
    --smoke)    SMOKE=1; shift ;;
    --epochs)   EPOCHS_OVERRIDE="$2"; shift 2 ;;
    --filter)   FILTER="$2"; shift 2 ;;
    --export)   EXPORT="$2"; shift 2 ;;
    --force)    FORCE=1; shift ;;
    --dry-run)  DRYRUN=1; shift ;;
    --bin)      BIN="$2"; shift 2 ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "[train-matrix] unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -x "$BIN" ] || { echo "[train-matrix] binary not found/executable: $BIN" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "[train-matrix] manifest not found: $MANIFEST" >&2; exit 2; }

# Resolve the dataset path for a task token.
data_for() {
  case "$1" in
    detect)   [ "$SMOKE" = 1 ] && echo "data/coco8"      || echo "${YOLOCPP_DATA_DETECT:-data/coco}" ;;
    segment)  [ "$SMOKE" = 1 ] && echo "data/coco8-seg"  || echo "${YOLOCPP_DATA_SEGMENT:-data/coco-seg}" ;;
    pose)     [ "$SMOKE" = 1 ] && echo "data/coco8-pose" || echo "${YOLOCPP_DATA_POSE:-data/coco-pose}" ;;
    obb)      [ "$SMOKE" = 1 ] && echo "data/dota8"      || echo "${YOLOCPP_DATA_OBB:-data/DOTAv1}" ;;
    classify) [ "$SMOKE" = 1 ] && echo "data"           || echo "${YOLOCPP_DATA_CLASSIFY:-data/imagenet}" ;;
    *) echo "" ;;
  esac
}

mkdir -p "$OUT"
RESULTS="$OUT/results.csv"
[ -f "$RESULTS" ] || echo "cell,task,scale,init,data,epochs,status,metric,seconds,artifact" > "$RESULTS"

n_run=0 n_skip=0 n_ok=0 n_fail=0
printf '[train-matrix] manifest=%s out=%s device=%s smoke=%s\n' "$MANIFEST" "$OUT" "$DEVICE" "$SMOKE"

# Manifest columns (tab-separated): cell  scale  task  init  epochs  imgsz  batch
while IFS=$'\t' read -r cell scale task init epochs imgsz batch _rest; do
  # Skip comments / blank lines.
  [ -z "${cell// /}" ] && continue
  case "$cell" in \#*) continue ;; esac
  [ -n "$FILTER" ] && ! echo "$cell" | grep -qE "$FILTER" && continue

  cell_dir="$OUT/$cell"
  # Detect trainers write best.pt; the task trainers write last.pt — either
  # means the cell is done.
  if { [ -f "$cell_dir/best.pt" ] || [ -f "$cell_dir/last.pt" ]; } && [ "$FORCE" != 1 ]; then
    echo "[skip] $cell — checkpoint already exists"; n_skip=$((n_skip+1)); continue
  fi

  data=$(data_for "$task")
  if [ -z "$data" ] || { [ ! -d "$data" ] && [ ! -f "$data" ]; }; then
    echo "[FAIL] $cell — dataset for task '$task' not found ($data)"
    echo "$cell,$task,$scale,$init,$data,-,nodata,NA,0,-" >> "$RESULTS"
    n_fail=$((n_fail+1)); continue
  fi

  ep="${EPOCHS_OVERRIDE:-$epochs}"; bt="$batch"
  if [ "$SMOKE" = 1 ]; then ep=2; bt=4; fi

  cmd=( "$BIN" --mode train --task "$task" --data "$data"
        --epochs "$ep" --imgsz "$imgsz" --batch "$bt"
        --device "$DEVICE" --scale "$scale" --save "$cell_dir" )
  [ "$init" != "scratch" ] && cmd+=( --model "$init" )
  [ -n "$EXPORT" ] && cmd+=( --export-after-train "$EXPORT" )

  if [ "$DRYRUN" = 1 ]; then
    printf '[dry] %s\n' "${cmd[*]}"; n_run=$((n_run+1)); continue
  fi

  # Fresh slate so --save lands in exactly cell_dir (the trainer auto-increments
  # an existing dir; a partial dir from an aborted run would otherwise skew it).
  rm -rf "$cell_dir"
  log="$OUT/${cell}.log"
  echo "[run ] $cell — task=$task scale=$scale init=$init data=$data epochs=$ep batch=$bt"
  t0=$(date +%s)
  "${cmd[@]}" >"$log" 2>&1; rc=$?
  t1=$(date +%s)
  n_run=$((n_run+1))

  artifact="$cell_dir/best.pt"; [ -f "$artifact" ] || artifact="$cell_dir/last.pt"
  if [ "$rc" = 0 ] && [ -f "$artifact" ]; then status=ok; else status=FAIL; fi

  # Metric: the detect trainer prints per-epoch val mAP; the task trainers don't.
  # Run a val pass on the saved checkpoint for a uniform final eval number
  # (detect mAP@0.5:0.95 / seg mask-mAP / pose OKS / obb rotated-mAP / cls top1).
  metric=""
  if [ "$status" = ok ]; then
    vlog="$OUT/${cell}.val.log"
    "$BIN" --mode val --task "$task" --model "$artifact" --data "$data" \
           --scale "$scale" >"$vlog" 2>&1 || true
    metric=$(grep -hoiE "mAP@0\.5:0\.95[ ]*=[ ]*[0-9.]+|mAP@0\.5[ ]*=[ ]*[0-9.]+|top1[ ]*=[ ]*[0-9.]+" \
             "$vlog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  fi

  if [ "$status" = ok ]; then
    n_ok=$((n_ok+1))
    echo "       ✓ $cell  metric=${metric:-NA}  ${artifact}  ($((t1-t0))s)"
  else
    n_fail=$((n_fail+1)); artifact="-"
    echo "       ✗ $cell  rc=$rc  (see $log)"
  fi
  echo "$cell,$task,$scale,$init,$data,$ep,$status,${metric:-NA},$((t1-t0)),$artifact" >> "$RESULTS"
done < "$MANIFEST"

echo "[train-matrix] done: run=$n_run ok=$n_ok fail=$n_fail skip=$n_skip  → $RESULTS"
[ "$n_fail" = 0 ]
