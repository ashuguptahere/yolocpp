#!/usr/bin/env bash
# Export ONNX for every (version, task, scale) combination into runs/onnx/.
# 3 versions × 5 tasks × 5 scales = 75 exports.
# Validation harness lives in /tmp/yolocpp_parity/validate_onnx*.py.

set -u
cd "$(dirname "$0")/.."

OUT="runs/onnx"
mkdir -p "${OUT}"
LOG="${OUT}/export.log"
: > "${LOG}"

# task → (suffix, default_imgsz)
declare -A TASK_SUFFIX=( [detect]="" [classify]="-cls" [segment]="-seg" \
                         [pose]="-pose" [obb]="-obb" )
declare -A TASK_IMGSZ=(  [detect]=640 [classify]=224 [segment]=640 \
                         [pose]=640   [obb]=1024 )

for v in 8 11 26; do
  for t in detect classify segment pose obb; do
    suf="${TASK_SUFFIX[${t}]}"
    imgsz="${TASK_IMGSZ[${t}]}"
    for s in n s m l x; do
      weight="data/yolo${v}${s}${suf}.pt"
      out_path="${OUT}/yolo${v}${s}${suf}.onnx"
      [[ -f "${weight}" ]] || { echo "SKIP ${weight} (missing)" >> "${LOG}"; continue; }
      rm -f "${out_path}"
      echo ">>> yolo${v}${s}${suf} → ${out_path}" | tee -a "${LOG}"
      set +e
      ./build/yolocpp task="${t}" mode=export model="${weight}" \
          format=onnx out="${out_path}" imgsz="${imgsz}" \
          >> "${LOG}" 2>&1
      rc=$?
      set -e
      if [[ -f "${out_path}" && ${rc} -eq 0 ]]; then
        sz=$(ls -lh "${out_path}" | awk '{print $5}')
        echo "    ok (${sz})" | tee -a "${LOG}"
      else
        echo "    FAIL (rc=${rc})" | tee -a "${LOG}"
      fi
    done
  done
done
echo
echo "exports written to ${OUT}/"
