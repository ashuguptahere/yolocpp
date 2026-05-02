#!/usr/bin/env bash
# Full (version × variant × task × mode) sweep.
#
# Closed set of supported YOLO versions (per CLAUDE.md):
#   yolo3 yolo4 yolo5 yolo6 yolo7 yolo8 yolo9 yolo10 yolo11 yolo12 yolo13 yolo26
# (yolo1, yolo2, yolo14..yolo25, yolo27+ are not supported and not planned.)
#
# Tasks: detect, segment, classify, pose, obb. Only v8/v11/v26 ship task
# weights upstream for all 5; v12/v13 ship detect-only; v3/v4/v5/v6/v7/v9/v10
# also ship detect-only (their task heads are not yet trained — tracked as
# future work).
#
# Modes: predict, val, train, export, benchmark. (No `test` mode in our CLI;
# `val` IS the test/eval pass.)
#
# This sweep:
#   - For every applicable (version, scale, task) cell, runs `predict` on
#     `data/bus.jpg` and `export format=onnx` against the local cache.
#   - For a representative subset, runs `train` (1 step, batch=1, coco8).
#   - For a representative subset, runs `val` on coco8.
#   - For a representative subset, runs `benchmark`.
#
# Output: pass/fail per cell + summary table at the end.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLI="$ROOT/build/yolocpp"
[ -x "$CLI" ] || { echo "[error] missing $CLI — run cmake --build build first"; exit 1; }

WEIGHTS="$HOME/.cache/yolocpp/weights"
WEIGHTS_ALT="$ROOT/data"
# Resolve a weight filename against both the cache and data/. Prints
# the first matching path or empty.
resolve_weight() {
  local name="$1"
  if [ -f "$WEIGHTS/$name" ]; then echo "$WEIGHTS/$name"; return; fi
  if [ -f "$WEIGHTS_ALT/$name" ]; then echo "$WEIGHTS_ALT/$name"; return; fi
  echo ""
}
SOURCE="$ROOT/data/bus.jpg"
DATA="$ROOT/data/coco8"
# Pass the directory to `--data` (cmd_val + cmd_train both expect a
# YOLO-layout dataset root). The yaml-resolution path lives in the
# kv-style code which was removed under #51K.
DATA_YAML="$DATA"
OUT="/tmp/yolocpp_full_sweep"
rm -rf "$OUT"; mkdir -p "$OUT"

LOG="$OUT/sweep.log"
RESULTS="$OUT/results.tsv"
echo -e "version\tscale\ttask\tmode\tstatus\tdetail" > "$RESULTS"

PASS=0; FAIL=0; SKIP=0

# Run a single CLI invocation under timeout, write status row.
run_one() {
  local ver="$1" sc="$2" task="$3" mode="$4" weights="$5" extra="${6:-}"
  if [ ! -f "$weights" ]; then
    echo -e "$ver\t$sc\t$task\t$mode\tSKIP\tweights_missing:$(basename "$weights")" >> "$RESULTS"
    SKIP=$((SKIP+1))
    return
  fi
  # All invocations use the canonical flag-style CLI (#51K removed
  # the kv-style and legacy subcommand parsers).
  local out_arg=""
  case "$mode" in
    predict)   out_arg="--out $OUT/${ver}${sc:+_$sc}_${task}_pred.jpg" ;;
    export)    out_arg="--format onnx --out $OUT/${ver}${sc:+_$sc}_${task}.onnx" ;;
    val)       out_arg="--data $DATA_YAML --scale $sc" ;;
    train)     out_arg="--data $DATA_YAML --scale $sc --epochs 1 --batch 1 --save $OUT/${ver}${sc:+_$sc}_${task}_train" ;;
    benchmark) out_arg="--source $SOURCE --warmup 1 --iters 1 --cache $OUT/bench_cache" ;;
  esac
  local cmd="$CLI --mode $mode --task $task --model $weights"
  case "$mode" in
    predict)   cmd="$cmd --source $SOURCE $out_arg $extra" ;;
    export)    cmd="$cmd $out_arg $extra" ;;
    val)       cmd="$cmd $out_arg $extra" ;;
    train)     cmd="$cmd $out_arg $extra" ;;
    # Benchmark auto-resolves version+scale from the weights filename
    # inside engine::run_benchmark, so we drop --task and --scale
    # here. (Bench is detect-only today.)
    benchmark) cmd="$CLI --mode benchmark --model $weights $out_arg $extra" ;;
  esac
  echo "==== $ver/$sc/$task/$mode ====" >> "$LOG"
  echo "$cmd" >> "$LOG"
  # Per-mode timeout: predict/export are quick; val/train/benchmark need
  # more headroom (val on coco8 = 4 batches × forward; train = 1 epoch).
  local tmo=90
  case "$mode" in val|train) tmo=300 ;; benchmark) tmo=180 ;; esac
  if timeout "$tmo" bash -c "$cmd" >>"$LOG" 2>&1; then
    echo -e "$ver\t$sc\t$task\t$mode\tPASS\t" >> "$RESULTS"
    PASS=$((PASS+1))
  else
    local rc=$?
    local last_err=$(tail -3 "$LOG" | tr '\n' ' ' | cut -c1-160)
    echo -e "$ver\t$sc\t$task\t$mode\tFAIL\trc=$rc:$last_err" >> "$RESULTS"
    FAIL=$((FAIL+1))
  fi
}

# ─── Per-version table ───────────────────────────────────────────────────
# Detect-only versions. Format: version → "scale1 scale2 ..." weight-pattern
# where weight pattern uses {SC} as placeholder for scale letter.
detect_only_versions() {
  # ver scales weight_template
  declare -A V
  V[v3]="-:yolo3.pt"
  V[v4]="-:yolo4.pt"
  # v5 uses the anchor-free `*u.pt` files (v8-style DFL head).
  V[v5]="n,s,m,l,x:yolo5{SC}u.pt"
  V[v6]="n,s,m,l,n6,s6,m6,l6,s_mbla,m_mbla,l_mbla,x_mbla:yolo6{SC}.pt"
  V[v7]="-,tiny,x,w6,e6,d6,e6e:yolo7{SC_DASH}.pt"
  V[v9]="t,s,m,c,e:yolo9{SC}.pt"
  # v10n is cached as `yolo10.pt` (no scale letter) — special-cased below.
  V[v10]="-,s,m,b,l,x:yolo10{SC}.pt"
  V[v12]="n,s,m,l,x:yolo12{SC}.pt"
  V[v13]="n,s,l,x:yolo13{SC}.pt"
  for ver in v3 v4 v5 v6 v7 v9 v10 v12 v13; do
    IFS=':' read -r scales tmpl <<< "${V[$ver]}"
    IFS=',' read -ra arr <<< "$scales"
    for sc in "${arr[@]}"; do
      local sc_arg="$sc" wtmpl="$tmpl"
      if [ "$sc" = "-" ]; then sc_arg=""; fi
      # v7's "-" weight is `yolo7.pt`; tiny/x/w6/e6/d6/e6e take `-tiny` etc.
      if [[ "$ver" == "v7" ]]; then
        if [ "$sc" = "-" ]; then wtmpl="${wtmpl/\{SC_DASH\}/}"
        elif [ "$sc" = "x" ]; then wtmpl="${wtmpl/\{SC_DASH\}/x}"
        else                         wtmpl="${wtmpl/\{SC_DASH\}/-$sc}"
        fi
      else
        wtmpl="${wtmpl/\{SC\}/$sc_arg}"
      fi
      # v10n: cached as `yolo10.pt` (no scale letter). When sc=- in the
      # v10 row, sc_arg is empty and wtmpl reads `yolo10.pt`; pass scale=n
      # to the CLI explicitly via `extra`.
      local extra_flags=""
      if [ "$ver" = "v10" ] && [ -z "$sc_arg" ]; then
        extra_flags="--scale n"
        sc_arg="n"
      fi
      local weights="$(resolve_weight "$wtmpl")"
      [ -z "$weights" ] && weights="$WEIGHTS/$wtmpl"  # leave for SKIP report
      run_one "$ver" "$sc_arg" "detect" "predict" "$weights" "$extra_flags"
    done
  done
}

# All-task versions (v8/v11/v26). Tasks: detect/cls/seg/pose/obb.
all_task_versions() {
  for ver in v8 v11 v26; do
    for sc in n s m l x; do
      for task in detect classify segment pose obb; do
        local suffix
        case "$task" in
          detect)   suffix="" ;;
          classify) suffix="-cls" ;;
          segment)  suffix="-seg" ;;
          pose)     suffix="-pose" ;;
          obb)      suffix="-obb" ;;
        esac
        local fname="yolo${ver#v}${sc}${suffix}.pt"
        local weights="$(resolve_weight "$fname")"
        [ -z "$weights" ] && weights="$WEIGHTS/$fname"
        run_one "$ver" "$sc" "$task" "predict" "$weights"
      done
    done
  done
}

echo "[sweep] phase 1: detect predict — every supported version × scale"
detect_only_versions
echo "[sweep] phase 2: all-task predict — v8/v11/v26 × n,s,m,l,x × 5 tasks"
all_task_versions

echo "[sweep] phase 3: export ONNX — one cell per version (smoke)"
for w in yolo3.pt yolo4.pt yolo5nu.pt yolo6n.pt yolo7.pt yolo8n.pt yolo9c.pt yolo10.pt yolo11n.pt yolo12n.pt yolo13n.pt yolo26n.pt; do
  ver=$(echo "$w" | grep -oE 'yolo[0-9]+' | sed 's/yolo/v/')
  weights="$(resolve_weight "$w")"; [ -z "$weights" ] && weights="$WEIGHTS/$w"
  run_one "$ver" "" "detect" "export" "$weights"
done

echo "[sweep] phase 4: smoke val (coco8) — one cell per version"
if [ -d "$DATA_YAML" ]; then
  for w in yolo5nu.pt yolo8n.pt yolo11n.pt yolo26n.pt; do
    ver=$(echo "$w" | grep -oE 'yolo[0-9]+' | sed 's/yolo/v/')
    weights="$(resolve_weight "$w")"; [ -z "$weights" ] && weights="$WEIGHTS/$w"
    run_one "$ver" "n" "detect" "val" "$weights"
  done
else
  echo -e "all\t-\tdetect\tval\tSKIP\tcoco8 missing" >> "$RESULTS"
  SKIP=$((SKIP+1))
fi

echo "[sweep] phase 5: smoke train (1 epoch, batch=1) — one cell per version"
if [ -d "$DATA_YAML" ] && [ -d "$DATA/images/train" ]; then
  for w in yolo8n.pt yolo11n.pt yolo26n.pt; do
    ver=$(echo "$w" | grep -oE 'yolo[0-9]+' | sed 's/yolo/v/')
    weights="$(resolve_weight "$w")"; [ -z "$weights" ] && weights="$WEIGHTS/$w"
    run_one "$ver" "n" "detect" "train" "$weights"
  done
else
  echo -e "all\t-\tdetect\ttrain\tSKIP\tcoco8 missing" >> "$RESULTS"
  SKIP=$((SKIP+1))
fi

echo "[sweep] phase 6: benchmark — every version (engine::run_benchmark now version-dispatched)"
# Benchmark covers all 12 versions via the new engine dispatch. We test
# the smallest variant of each (fastest TRT engine build). v5 uses the
# u-form anchor-free file; v10n is cached as `yolo10.pt` (no scale letter).
for w in yolo3.pt yolo4.pt yolo5nu.pt yolo6n.pt yolo7-tiny.pt yolo8n.pt \
         yolo9t.pt yolo10.pt yolo11n.pt yolo12n.pt yolo13n.pt yolo26n.pt; do
  ver=$(echo "$w" | grep -oE 'yolo[0-9]+' | sed 's/yolo/v/')
  weights="$(resolve_weight "$w")"; [ -z "$weights" ] && weights="$WEIGHTS/$w"
  run_one "$ver" "n" "detect" "benchmark" "$weights"
done

# ─── Phase 7: TRT round-trip per version (#53B). ─────────────────────────
# For every version that has a cached `.pt`, build a `.trt` engine
# (FP16) via `--mode export` and then run `--mode predict` against the
# resulting engine. Catches any `.pt` → `.onnx` → `.trt` regression
# the per-version export emitter, the TensorRT parser, or the runtime
# might introduce. Smoke-only — strict numerical parity is the job of
# `tests/test_cross_backend_parity.cpp` (#53A).
echo "[sweep] phase 7: TRT export+predict round-trip — every version"
TRT_OUT="$OUT/trt_round_trip"
mkdir -p "$TRT_OUT"
for w in yolo3.pt yolo4.pt yolo5nu.pt yolo6n.pt yolo7-tiny.pt yolo8n.pt \
         yolo9t.pt yolo10.pt yolo11n.pt yolo12n.pt yolo13n.pt yolo26n.pt; do
  ver=$(echo "$w" | grep -oE 'yolo[0-9]+' | sed 's/yolo/v/')
  weights="$(resolve_weight "$w")"; [ -z "$weights" ] && weights="$WEIGHTS/$w"
  if [ ! -f "$weights" ]; then
    echo -e "$ver\tn\tdetect\ttrt-roundtrip\tSKIP\tweights_missing" >> "$RESULTS"
    SKIP=$((SKIP+1))
    continue
  fi
  trt_out="$TRT_OUT/${ver}.trt"
  pred_jpg="$TRT_OUT/${ver}_trt_pred.jpg"
  # 1. Export to .trt (FP16). We pass --scale n (with the v10
  # exception baked into run_one's extra_flags style) to keep imgsz
  # and channel widths consistent with the cached weights.
  extra=""
  case "$ver" in v10) extra="--scale n" ;; esac
  cmd="$CLI --mode export --task detect --model $weights --format trt --out $trt_out $extra"
  echo "==== $ver/n/detect/trt-roundtrip-export ====" >> "$LOG"
  echo "$cmd" >> "$LOG"
  if ! timeout 300 bash -c "$cmd" >>"$LOG" 2>&1; then
    echo -e "$ver\tn\tdetect\ttrt-roundtrip\tFAIL\texport_failed" >> "$RESULTS"
    FAIL=$((FAIL+1))
    continue
  fi
  # 2. Predict against the engine. detect-task TRT routes through
  # `inference::TrtPredictor` regardless of source version.
  cmd="$CLI --mode predict --task detect --model $trt_out --source $SOURCE --out $pred_jpg"
  echo "==== $ver/n/detect/trt-roundtrip-predict ====" >> "$LOG"
  echo "$cmd" >> "$LOG"
  if timeout 60 bash -c "$cmd" >>"$LOG" 2>&1; then
    echo -e "$ver\tn\tdetect\ttrt-roundtrip\tPASS\t" >> "$RESULTS"
    PASS=$((PASS+1))
  else
    echo -e "$ver\tn\tdetect\ttrt-roundtrip\tFAIL\tpredict_failed" >> "$RESULTS"
    FAIL=$((FAIL+1))
  fi
done

echo
echo "================================================================"
echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP   ($(wc -l < "$RESULTS") rows)"
echo "================================================================"
echo "[sweep] details: $RESULTS"
echo "[sweep] full log: $LOG"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
