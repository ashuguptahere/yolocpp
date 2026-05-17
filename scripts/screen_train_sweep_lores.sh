#!/usr/bin/env bash
# Low-resolution variant of the sweep — flushes out any hardcoded
# imgsz / feature-map / anchor-cell-count assumptions that the
# registry-default-imgsz path doesn't exercise.
#
# Targets:
#   v1:                   imgsz=448  (FC head won't go lower)
#   v2 (full+tiny):       imgsz=320  (stride-32 → 10×10 head)
#   v6-P6 / v7-P6:        imgsz=384  (stride-64 → 6×6 deep head)
#   everything else (P5): imgsz=320
set -u

DATA="/home/ashu/Desktop/dataset/screen-dataset-yolo/data.yaml"
OUT_ROOT="/tmp/screen_train_sweep_lores_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
CSV="${OUT_ROOT}/RESULTS.csv"
echo "variant,weights,imgsz,result,final_loss,val_map_50_95,log" > "${CSV}"
YOLOCPP="/home/ashu/Desktop/yolocpp/build/yolocpp"

# (name, weights, batch, imgsz)
VARIANTS=(
  "yolo1               yolo1                       4   448"
  "yolo2               data/yolo2.pt               8   320"
  "yolo2-tiny          data/yolo2-tiny-voc.pt      8   320"
  "yolo3u              yolo3u.pt                   8   320"
  "yolo4               data/yolo4.pt               8   320"
  "yolo5n              data/yolo5n.pt              8   320"
  "yolo5s              data/yolo5s.pt              8   320"
  "yolo5m              data/yolo5m.pt              8   320"
  "yolo5l              data/yolo5l.pt              8   320"
  "yolo5x              data/yolo5x.pt              4   320"
  "yolo6n              yolo6n.pt                   8   320"
  "yolo6s              yolo6s.pt                   8   320"
  "yolo6m              yolo6m.pt                   8   320"
  "yolo6l              yolo6l.pt                   8   320"
  "yolo6s_mbla         yolo6s_mbla.pt              8   320"
  "yolo6m_mbla         yolo6m_mbla.pt              8   320"
  "yolo6l_mbla         yolo6l_mbla.pt              8   320"
  "yolo6x_mbla         yolo6x_mbla.pt              4   320"
  "yolo6n6             yolo6n6.pt                  8   384"
  "yolo6s6             yolo6s6.pt                  8   384"
  "yolo6m6             yolo6m6.pt                  8   384"
  "yolo6l6             yolo6l6.pt                  4   384"
  "yolo7               yolo7.pt                    8   320"
  "yolo7-tiny          yolo7-tiny.pt               8   320"
  "yolo7x              yolo7x.pt                   4   320"
  "yolo8n              data/yolo8n.pt              8   320"
  "yolo8s              data/yolo8s.pt              8   320"
  "yolo8m              data/yolo8m.pt              8   320"
  "yolo8l              data/yolo8l.pt              8   320"
  "yolo8x              data/yolo8x.pt              4   320"
  "yolo9t              yolo9t.pt                   8   320"
  "yolo9s              yolo9s.pt                   8   320"
  "yolo9m              yolo9m.pt                   8   320"
  "yolo9c              yolo9c.pt                   8   320"
  "yolo9e              yolo9e.pt                   4   320"
  "yolo10n             yolo10n.pt                  8   320"
  "yolo10s             yolo10s.pt                  8   320"
  "yolo10m             yolo10m.pt                  8   320"
  "yolo10b             yolo10b.pt                  8   320"
  "yolo10l             yolo10l.pt                  8   320"
  "yolo10x             yolo10x.pt                  4   320"
  "yolo11n             data/yolo11n.pt             8   320"
  "yolo11s             data/yolo11s.pt             8   320"
  "yolo11m             data/yolo11m.pt             8   320"
  "yolo11l             data/yolo11l.pt             8   320"
  "yolo11x             data/yolo11x.pt             4   320"
  "yolo12n             data/yolo12n.pt             8   320"
  "yolo12s             data/yolo12s.pt             8   320"
  "yolo12m             data/yolo12m.pt             8   320"
  "yolo12l             data/yolo12l.pt             8   320"
  "yolo12x             data/yolo12x.pt             4   320"
  "yolo13n             data/yolo13n.pt             8   320"
  "yolo13s             data/yolo13s.pt             8   320"
  "yolo13l             data/yolo13l.pt             8   320"
  "yolo13x             data/yolo13x.pt             4   320"
  "yolo26n             data/yolo26n.pt             8   320"
  "yolo26s             data/yolo26s.pt             8   320"
  "yolo26m             data/yolo26m.pt             8   320"
  "yolo26l             data/yolo26l.pt             8   320"
  "yolo26x             data/yolo26x.pt             4   320"
)

run_one() {
  local name="$1" weights="$2" batch="$3" imgsz="$4"
  local logfile="${OUT_ROOT}/${name}.log"
  local save_dir="${OUT_ROOT}/${name}_train"
  echo "=== ${name} (m=${weights}, batch=${batch}, imgsz=${imgsz}) ===" | tee -a "${logfile}"

  if timeout 300 "${YOLOCPP}" --mode train -m "${weights}" -d "${DATA}" \
        --epochs 1 --batch "${batch}" --imgsz "${imgsz}" \
        --save "${save_dir}" --seed 42 \
        >> "${logfile}" 2>&1; then
    local loss
    loss=$(grep -E "epoch 0 avg total=" "${logfile}" | tail -1 |
             sed -E 's/.*total=([0-9.e+-]+).*/\1/')
    local m
    m=$(grep "mAP@0.5:0.95=" "${logfile}" | tail -1 |
          sed -E 's/.*mAP@0\.5:0\.95=([0-9.e+-]+).*/\1/')
    echo "${name},${weights},${imgsz},PASS,${loss:-NA},${m:-NA},${logfile}" >> "${CSV}"
    echo "  PASS ${name}@${imgsz} loss=${loss:-NA} map=${m:-NA}"
  else
    local rc=$?
    echo "${name},${weights},${imgsz},FAIL(rc=${rc}),NA,NA,${logfile}" >> "${CSV}"
    echo "  FAIL ${name}@${imgsz} rc=${rc}"
  fi
}

for line in "${VARIANTS[@]}"; do
  read -r name weights batch imgsz <<< "${line}"
  run_one "${name}" "${weights}" "${batch}" "${imgsz}"
done

echo
echo "=== DONE ==="
wc -l "${CSV}"
