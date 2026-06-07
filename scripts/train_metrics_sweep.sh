#!/usr/bin/env bash
# 1-epoch training-metrics sweep. For every supported (version, variant),
# train ONE epoch on the screen-detection dataset and record:
#   mAP@0.5, mAP@0.5:0.95, precision, recall, F1   (from the trainer's val)
#   train CPU%, peak RSS (RAM), peak VRAM           (resource usage)
# Weights resolve + download into ./models/ (source URL printed on fetch).
# Output: a per-variant CSV. Pair with tools to diff vs docs/data/training.csv.
#
# Usage: scripts/train_metrics_sweep.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:${PATH}"

BIN="${ROOT}/build/yolocpp"
[ -x "$BIN" ] || { echo "[error] missing $BIN — build first"; exit 1; }
DATA="${DATA:-/home/ashu/Desktop/dataset/screen-dataset-yolo/data.yaml}"
[ -f "$DATA" ] || { echo "[error] dataset not found: $DATA"; exit 1; }

OUT="${OUT:-/tmp/train_metrics_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
CSV="${OUT}/metrics.csv"
echo "variant,mAP50,mAP,P,R,F1,CPU_pct,RSS_GB,VRAM_MB,batch,train_sec,status" > "$CSV"
echo "[sweep] output → $OUT"

# Reuse the canonical 60-variant list from screen_train_sweep.sh (no dup).
mapfile -t VARIANTS < <(sed -n '/VARIANTS=(/,/^)/p' scripts/screen_train_sweep.sh \
  | grep -E '^[[:space:]]*"' | sed -E 's/^[[:space:]]*"//; s/"[[:space:]]*$//')
echo "[sweep] ${#VARIANTS[@]} variants, 1 epoch each on $(basename "$(dirname "$DATA")")"

run_one() {
  local name="$1" weights="$2" batch="$3"
  local save="${OUT}/${name}" tlog="${OUT}/${name}.train.log"
  local timef="${OUT}/${name}.time.txt" vramf="${OUT}/${name}.vram.txt"
  : > "$vramf"
  # Peak-VRAM sampler (1 Hz) for the duration of this variant's training.
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        2>/dev/null >> "$vramf"; sleep 1; done ) & local sampler=$!
  local t0; t0=$(date +%s)
  /usr/bin/time -v -o "$timef" timeout 1200 "$BIN" --mode train -m "$weights" -d "$DATA" \
      --epochs 1 --batch "$batch" --imgsz 640 --seed 42 --save "$save" > "$tlog" 2>&1
  local rc=$?
  local t1; t1=$(date +%s)
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null

  local vline; vline=$(grep -E '\[trainer\] val:' "$tlog" | tail -1)
  local map50 map p r f1
  map50=$(grep -oE 'mAP@0\.5=[0-9.]+'      <<<"$vline" | head -1 | cut -d= -f2)
  map=$(  grep -oE 'mAP@0\.5:0\.95=[0-9.]+'<<<"$vline" | cut -d= -f2)
  p=$(    grep -oE ' P=[0-9.]+'            <<<"$vline" | tr -d ' ' | cut -d= -f2)
  r=$(    grep -oE ' R=[0-9.]+'            <<<"$vline" | tr -d ' ' | cut -d= -f2)
  f1=$(   grep -oE ' F1=[0-9.]+'           <<<"$vline" | tr -d ' ' | cut -d= -f2)

  local cpu rss_kb rss_gb vram
  cpu=$(grep 'Percent of CPU' "$timef" 2>/dev/null | grep -oE '[0-9]+%' | tr -d '%')
  rss_kb=$(grep 'Maximum resident set size' "$timef" 2>/dev/null | grep -oE '[0-9]+$')
  rss_gb=$(awk "BEGIN{printf \"%.1f\", ${rss_kb:-0}/1048576}")
  vram=$(sort -n "$vramf" 2>/dev/null | tail -1)

  local status=ok
  [ "$rc" -ne 0 ] && status="rc=${rc}"
  echo "${name},${map50},${map},${p},${r},${f1},${cpu},${rss_gb},${vram},${batch},$((t1-t0)),${status}" >> "$CSV"
  printf '[%-12s] mAP50=%-7s mAP=%-7s P=%-7s R=%-7s | CPU=%s%% RSS=%sG VRAM=%sM | %ss rc=%s\n' \
      "$name" "${map50:-NA}" "${map:-NA}" "${p:-NA}" "${r:-NA}" "${cpu:-NA}" "${rss_gb:-NA}" "${vram:-NA}" "$((t1-t0))" "$rc"
}

for line in "${VARIANTS[@]}"; do
  read -r name weights batch <<< "$line"
  run_one "$name" "$weights" "$batch"
done

echo "=== DONE → $CSV ==="
column -t -s, "$CSV"
