#!/usr/bin/env bash
# rfdetr variant sweep — runs after the YOLO sweep finishes (waits for the
# GPU to free up by polling for any other yolocpp train process).
set -u
DATA=/home/ashu/Desktop/dataset/screen-dataset-yolo/data.yaml
OUT=/tmp/screen_sweep
RESULTS=$OUT/RESULTS.csv
EPOCHS=${EPOCHS:-2}   # rfdetr is heavier; 2 epochs is enough to see a trend.
mkdir -p $OUT runs/train/sweep

run() {
  local name=$1 model=$2 imgsz=${3:-640} batch=${4:-4}
  local log=$OUT/$name.log
  local save=runs/train/sweep/$name
  rm -rf $save
  echo "=== $name (model=$model imgsz=$imgsz batch=$batch) ===" | tee -a $OUT/sweep.log
  local t0=$(date +%s)
  stdbuf -oL -eL ~/Desktop/yolocpp/build/yolocpp --mode train \
    -m "$model" -d "$DATA" -e $EPOCHS -b $batch --imgsz $imgsz \
    -D cuda --seed 42 --save "$save" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s); local sec=$(( t1 - t0 ))
  local sec_ep=$(( sec / (EPOCHS > 0 ? EPOCHS : 1) ))
  local best50=$(grep "val: mAP@0.5=" "$log" | awk -F'mAP@0.5=' '{print $2}' | awk '{print $1}' | sort -gr | head -1)
  local best5095=$(grep "val: mAP@0.5=" "$log" | awk -F'mAP@0.5:0.95=' '{print $2}' | awk '{print $1}' | sort -gr | head -1)
  best50=${best50:-NA}; best5095=${best5095:-NA}
  local note=""; [ $rc -ne 0 ] && note="exit$rc"
  echo "$name,$EPOCHS,$batch,$imgsz,$rc,$best50,$best5095,$sec_ep,$note" >> $RESULTS
  echo "  -> rc=$rc best50=$best50 best5095=$best5095 ${sec_ep}s/ep" | tee -a $OUT/sweep.log
}

# Wait for any other yolocpp training process to exit first.
echo "[rfdetr-sweep] waiting for GPU..." | tee -a $OUT/sweep.log
until ! pgrep -f "yolocpp --mode train" >/dev/null; do sleep 30; done

run rfdetr_n  rf-detr-nano.pth   640 4
run rfdetr_s  rf-detr-small.pth  640 4
run rfdetr_m  rf-detr-medium.pth 640 4
run rfdetr_b  rf-detr-base.pth   640 2
run rfdetr_l  rf-detr-large.pth  640 2

echo "=== rfdetr sweep done ==="
