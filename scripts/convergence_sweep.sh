#!/usr/bin/env bash
# Multi-version × multi-scale convergence sweep on coco8.
# For each (version, scale) pair: 15-epoch fine-tune from the official
# upstream .pt; capture first-epoch total loss, last-epoch total loss,
# best val mAP@0.5:0.95.
#
# Output: runs/convergence/<version>_<scale>/{stdout.log, results.csv, summary}
# A combined CSV summary is written to runs/convergence/summary.csv.

set -u
cd "$(dirname "$0")/.."

DATA="data/coco8/data.yaml"
EPOCHS=15
BATCH=4
IMGSZ=640
ROOT="runs/convergence"
SUMMARY="${ROOT}/summary.csv"

mkdir -p "${ROOT}"
echo "version,scale,first_total,last_total,best_map_5095,status" > "${SUMMARY}"

# (version_token, weight_template). yolo5 ships as yolov5<s>u.pt upstream;
# the auto-resolver maps the canonical local name to the upstream URL.
versions=(
  "yolo5"
  "yolo8"
  "yolo11"
  "yolo26"
)

for v in "${versions[@]}"; do
  for s in n s m l x; do
    weight="${v}${s}.pt"
    # yolo5 ships as `*u.pt` (anchorless); use the canonical alias.
    if [[ "$v" == "yolo5" ]]; then
      weight="yolo5${s}u.pt"
    fi
    tag="${v}_${s}"
    out="${ROOT}/${tag}"
    mkdir -p "${out}"
    log="${out}/stdout.log"
    echo ">>> [${tag}] training: model=${weight} epochs=${EPOCHS} batch=${BATCH}"

    set +e
    ./build/yolocpp task=detect mode=train model="${weight}" \
      data="${DATA}" epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} \
      save="${out}/run" patience=0 \
      > "${log}" 2>&1
    rc=$?
    set -e

    # The trainer writes results.csv to <save>/results.csv. Find it (the
    # save_dir auto-increments if it already existed).
    res_csv=$(ls -1d "${out}"/run*/results.csv 2>/dev/null | head -1)
    first_total="" ; last_total="" ; best_map=""
    if [[ -n "${res_csv}" && -f "${res_csv}" ]]; then
      # results.csv columns: epoch,time,box,cls,dfl,map50,map5095,lr0
      first_total=$(awk -F, 'NR==2{printf "%.4f", $3+$4+$5}' "${res_csv}")
      last_total=$(awk -F,  'END{printf "%.4f", $3+$4+$5}' "${res_csv}")
      best_map=$(awk -F,    'NR>1 && $7>m{m=$7} END{printf "%.4f", m+0}' "${res_csv}")
    fi
    status="ok"
    if [[ ${rc} -ne 0 ]]; then status="rc=${rc}"; fi
    echo "${v},${s},${first_total},${last_total},${best_map},${status}" \
        | tee -a "${SUMMARY}"
  done
done

echo
echo "=== Final summary (${SUMMARY}) ==="
column -s, -t "${SUMMARY}"
