#!/usr/bin/env bash
# Train every supported (version, variant) for 1 epoch on the
# screen-detection dataset, capture PASS/FAIL + final loss + val mAP,
# write a CSV summary. Designed to be background-run.
#
# Each run uses:
#   --epochs 1, --batch 8 (drops to 4 for x-scale to avoid OOM),
#   --imgsz from the registry default (no override).

set -u

DATA="/home/ashu/Desktop/dataset/screen-dataset-yolo/data.yaml"
OUT_ROOT="/tmp/screen_train_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
CSV="${OUT_ROOT}/RESULTS.csv"
echo "variant,weights,result,final_loss,val_map_50_95,log" > "${CSV}"

YOLOCPP="/home/ashu/Desktop/yolocpp/build/yolocpp"

# Every (variant_name, weights_arg, batch_size) tuple. variant_name is
# the identifier we'll use in the CSV; weights_arg is what we pass to
# `-m` (a bare name → train from scratch; a .pt → pretrained init).
VARIANTS=(
  # Darknet-era
  "yolo1               yolo1                                              8"
  "yolo2               data/yolo2.pt                                      8"
  "yolo2-tiny          data/yolo2-tiny-voc.pt                             8"
  # v3 / v4 (Darknet)
  "yolo3u              yolo3u.pt                                          8"
  "yolo4               data/yolo4.pt                                      8"
  # v5
  "yolo5n              data/yolo5n.pt                                     8"
  "yolo5s              data/yolo5s.pt                                     8"
  "yolo5m              data/yolo5m.pt                                     8"
  "yolo5l              data/yolo5l.pt                                     8"
  "yolo5x              data/yolo5x.pt                                     4"
  # v6 P5
  "yolo6n              yolo6n.pt                                          8"
  "yolo6s              yolo6s.pt                                          8"
  "yolo6m              yolo6m.pt                                          8"
  "yolo6l              yolo6l.pt                                          8"
  # v6 MBLA
  "yolo6s_mbla         yolo6s_mbla.pt                                     8"
  "yolo6m_mbla         yolo6m_mbla.pt                                     8"
  "yolo6l_mbla         yolo6l_mbla.pt                                     8"
  "yolo6x_mbla         yolo6x_mbla.pt                                     4"
  # v6 P6 — imgsz=1280; small batches to stay under 32 GB.
  "yolo6n6             yolo6n6.pt                                         4"
  "yolo6s6             yolo6s6.pt                                         4"
  "yolo6m6             yolo6m6.pt                                         4"
  "yolo6l6             yolo6l6.pt                                         2"
  # v7 (base/tiny/x — P6 variants segfault per CLAUDE.md, skipped)
  "yolo7               yolo7.pt                                           8"
  "yolo7-tiny          yolo7-tiny.pt                                      8"
  "yolo7x              yolo7x.pt                                          4"
  # v8
  "yolo8n              data/yolo8n.pt                                     8"
  "yolo8s              data/yolo8s.pt                                     8"
  "yolo8m              data/yolo8m.pt                                     8"
  "yolo8l              data/yolo8l.pt                                     8"
  "yolo8x              data/yolo8x.pt                                     4"
  # v9
  "yolo9t              yolo9t.pt                                          8"
  "yolo9s              yolo9s.pt                                          8"
  "yolo9m              yolo9m.pt                                          8"
  "yolo9c              yolo9c.pt                                          8"
  "yolo9e              yolo9e.pt                                          4"
  # v10
  "yolo10n             yolo10n.pt                                         8"
  "yolo10s             yolo10s.pt                                         8"
  "yolo10m             yolo10m.pt                                         8"
  "yolo10b             yolo10b.pt                                         8"
  "yolo10l             yolo10l.pt                                         8"
  "yolo10x             yolo10x.pt                                         4"
  # v11
  "yolo11n             data/yolo11n.pt                                    8"
  "yolo11s             data/yolo11s.pt                                    8"
  "yolo11m             data/yolo11m.pt                                    8"
  "yolo11l             data/yolo11l.pt                                    8"
  "yolo11x             data/yolo11x.pt                                    4"
  # v12
  "yolo12n             data/yolo12n.pt                                    8"
  "yolo12s             data/yolo12s.pt                                    8"
  "yolo12m             data/yolo12m.pt                                    8"
  "yolo12l             data/yolo12l.pt                                    8"
  "yolo12x             data/yolo12x.pt                                    4"
  # v13 (no m upstream)
  "yolo13n             data/yolo13n.pt                                    8"
  "yolo13s             data/yolo13s.pt                                    8"
  "yolo13l             data/yolo13l.pt                                    8"
  "yolo13x             data/yolo13x.pt                                    4"
  # v26
  "yolo26n             data/yolo26n.pt                                    8"
  "yolo26s             data/yolo26s.pt                                    8"
  "yolo26m             data/yolo26m.pt                                    8"
  "yolo26l             data/yolo26l.pt                                    8"
  "yolo26x             data/yolo26x.pt                                    4"
)

run_one() {
  local name="$1" weights="$2" batch="$3"
  local logfile="${OUT_ROOT}/${name}.log"
  local save_dir="${OUT_ROOT}/${name}_train"
  echo "=== ${name} (m=${weights}, batch=${batch}) ===" | tee -a "${logfile}"

  if timeout 600 "${YOLOCPP}" --mode train -m "${weights}" -d "${DATA}" \
        --epochs 1 --batch "${batch}" --save "${save_dir}" \
        --seed 42 \
        >> "${logfile}" 2>&1; then
    local loss
    loss=$(grep -E "epoch 0 avg total=" "${logfile}" | tail -1 |
             sed -E 's/.*total=([0-9.e+-]+).*/\1/')
    local m
    m=$(grep "mAP@0.5:0.95=" "${logfile}" | tail -1 |
          sed -E 's/.*mAP@0\.5:0\.95=([0-9.e+-]+).*/\1/')
    echo "${name},${weights},PASS,${loss:-NA},${m:-NA},${logfile}" >> "${CSV}"
    echo "  ✅ ${name} loss=${loss:-NA} mAP@.5:.95=${m:-NA}"
  else
    local rc=$?
    echo "${name},${weights},FAIL(rc=${rc}),NA,NA,${logfile}" >> "${CSV}"
    echo "  ❌ ${name} FAIL (rc=${rc}); see ${logfile}"
  fi
}

for line in "${VARIANTS[@]}"; do
  read -r name weights batch <<< "${line}"
  run_one "${name}" "${weights}" "${batch}"
done

echo
echo "=== Summary ==="
echo "results: ${CSV}"
column -ts, "${CSV}" | head -64
