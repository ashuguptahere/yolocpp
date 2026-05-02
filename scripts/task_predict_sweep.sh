#!/usr/bin/env bash
# Predict-path smoke test for every (version, task, scale) combination.
# Just runs `./build/yolocpp ... mode=predict source=data/bus.jpg`
# and records exit code + a short status line per run. Failures are
# either weight-load mismatches or runtime forward errors.

set -u
cd "$(dirname "$0")/.."

ROOT="runs/task_smoke"
mkdir -p "${ROOT}"
SUMMARY="${ROOT}/summary.csv"
echo "version,task,scale,status,detail" > "${SUMMARY}"

SOURCE="data/bus.jpg"

run_one() {
  local v="$1" t="$2" s="$3" weight="$4" img="$5"
  local tag="yolo${v}${s}-${t}"
  local log="${ROOT}/${tag}.log"
  set +e
  ./build/yolocpp task="${t}" mode=predict model="${weight}" \
      source="${img}" out="${ROOT}/${tag}.jpg" \
      > "${log}" 2>&1
  local rc=$?
  set -e
  local detail=""
  if [[ ${rc} -eq 0 ]]; then
    detail=$(grep -oE '\[(predict|classify|segment|pose|obb)\][^"]*detection|instance|people|rotated|top-5' "${log}" | head -1)
    [[ -z "${detail}" ]] && detail=$(tail -1 "${log}")
    echo "${v},${t},${s},ok,\"${detail}\"" | tee -a "${SUMMARY}"
  else
    detail=$(tail -1 "${log}" | tr -d '"')
    echo "${v},${t},${s},FAIL,\"${detail}\"" | tee -a "${SUMMARY}"
  fi
}

# v8 / v11 / v26 × {detect, classify, segment, pose, obb} × {n, s, m, l, x}
for v in 8 11 26; do
  for t in detect classify segment pose obb; do
    for s in n s m l x; do
      if [[ "${t}" == "detect" ]]; then
        weight="data/yolo${v}${s}.pt"
      else
        # Upstream suffix pattern: -cls / -seg / -pose / -obb.
        suffix=$(echo "${t}" | sed 's/classify/cls/; s/segment/seg/')
        weight="data/yolo${v}${s}-${suffix}.pt"
      fi
      [[ -f "${weight}" ]] || { echo "${v},${t},${s},SKIP,missing-weight" | tee -a "${SUMMARY}"; continue; }
      img="${SOURCE}"
      run_one "${v}" "${t}" "${s}" "${weight}" "${img}"
    done
  done
done

echo
echo "=== task predict smoke summary ==="
column -s, -t "${SUMMARY}"
