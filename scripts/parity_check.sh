#!/usr/bin/env bash
# Measure mAP@0.5 / mAP@0.5:0.95 on full COCO val (5000 images) for every
# yolo11 and yolo26 scale, using only the Ultralytics-shipped weights (no
# fine-tune). The intent is to compare against Ultralytics' published
# numbers and quantify the parity gap.

set -u
cd "$(dirname "$0")/.."

DATA="data/coco/data.yaml"
ROOT="runs/parity"
SUMMARY="${ROOT}/summary.csv"
mkdir -p "${ROOT}"
echo "version,scale,map_50,map_50_95,status" > "${SUMMARY}"

run_one() {
  local version="$1" scale="$2" weight="$3"
  local tag="${version}_${scale}"
  local log="${ROOT}/${tag}.log"
  set +e
  ./build/yolocpp task=detect mode=val model="${weight}" data="${DATA}" \
      > "${log}" 2>&1
  local rc=$?
  set -e
  local m50=$(grep "mAP@0.5 " "${log}" | awk '{print $NF}')
  local m5095=$(grep "mAP@0.5:0.95" "${log}" | awk '{print $NF}')
  local status="ok"; [[ ${rc} -ne 0 ]] && status="rc=${rc}"
  echo "${version},${scale},${m50},${m5095},${status}" | tee -a "${SUMMARY}"
}

for s in n s m l x; do
  run_one yolo11 "${s}" "/home/ashu/Desktop/yolocpp/data/yolo11${s}.pt"
done
for s in n s m l x; do
  run_one yolo26 "${s}" "/home/ashu/Desktop/yolocpp/yolo26${s}.pt"
done

echo
echo "=== parity summary ==="
column -s, -t "${SUMMARY}"
