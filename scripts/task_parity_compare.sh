#!/usr/bin/env bash
# Side-by-side compare of our predict outputs vs Ultralytics Python.
# Reads runs/task_smoke/*.log (produced by task_predict_sweep.sh) and
# extracts (count, top-conf) per (version, task, scale).
#
# Python ground-truth values come from the table dumped by
# /tmp/yolocpp_parity/validate_tasks.py (run that first).

set -u
cd "$(dirname "$0")/.."

ROOT="runs/task_smoke"
echo "version  task       scale  ours             python (manual reference)"
for v in 8 11 26; do
  for t in detect classify segment pose obb; do
    for s in n s; do
      tag="yolo${v}${s}-${t}"
      log="${ROOT}/${tag}.log"
      [[ -f "${log}" ]] || continue
      ours=""
      case "${t}" in
        detect|"")
          ours=$(grep -oE '[0-9]+ detections' "${log}" | head -1) ;;
        segment)
          ours=$(grep -oE '[0-9]+ instances' "${log}" | head -1) ;;
        pose)
          ours=$(grep -oE '[0-9]+ people' "${log}" | head -1) ;;
        obb)
          ours=$(grep -oE '[0-9]+ rotated boxes' "${log}" | head -1) ;;
        classify)
          ours=$(grep -A1 'top-5:' "${log}" | tail -1 | tr -s ' ')
          ours="top1=${ours}"
          ;;
      esac
      printf "%-7s  %-9s  %-5s  %-15s  (see Python table)\n" "${v}" "${t}" "${s}" "${ours}"
    done
  done
done
