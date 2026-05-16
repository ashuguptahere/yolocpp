#!/usr/bin/env bash
# Train every (version, variant) cell on the screen dataset and tabulate mAP.
#
# - Uses 3 epochs (enough to see useful trends; gets ~mAP 0.5+ on most
#   variants).
# - Per-variant batch size is downscaled for the heavier scales.
# - Output: /tmp/screen_sweep/<variant>.log + /tmp/screen_sweep/RESULTS.csv

set -u
DATA=/home/ashu/Desktop/dataset/screen-dataset-yolo/data.yaml
OUT=/tmp/screen_sweep
RESULTS=$OUT/RESULTS.csv
EPOCHS=${EPOCHS:-3}
mkdir -p $OUT runs/train/sweep
echo "variant,epochs,batch,imgsz,exit,best_map50,best_map5095,sec_per_ep,note" > $RESULTS

run() {
  local name=$1 model=$2 imgsz=${3:-640} batch=${4:-16}
  local log=$OUT/$name.log
  local save=runs/train/sweep/$name
  rm -rf $save
  echo "=== $name (model=$model imgsz=$imgsz batch=$batch) ===" | tee -a $OUT/sweep.log
  local t0=$(date +%s)
  stdbuf -oL -eL ~/Desktop/yolocpp/build/yolocpp --mode train \
    -m "$model" -d "$DATA" -e $EPOCHS -b $batch --imgsz $imgsz \
    -D cuda --seed 42 --save "$save" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local sec=$(( t1 - t0 ))
  local sec_ep=$(( sec / (EPOCHS > 0 ? EPOCHS : 1) ))
  # Extract best mAP@0.5 & mAP@0.5:0.95 across epochs
  local best50=$(grep "val: mAP@0.5=" "$log" | awk -F'mAP@0.5=' '{print $2}' | awk '{print $1}' | sort -gr | head -1)
  local best5095=$(grep "val: mAP@0.5=" "$log" | awk -F'mAP@0.5:0.95=' '{print $2}' | awk '{print $1}' | sort -gr | head -1)
  best50=${best50:-NA}; best5095=${best5095:-NA}
  local note=""
  if [ $rc -ne 0 ]; then note="exit$rc"; fi
  echo "$name,$EPOCHS,$batch,$imgsz,$rc,$best50,$best5095,$sec_ep,$note" >> $RESULTS
  echo "  -> rc=$rc best50=$best50 best5095=$best5095 ${sec_ep}s/ep" | tee -a $OUT/sweep.log
}

# v3 — only u-form
run v3u  yolo3u.pt    640 16

# v4 — single scale (608² calibrated anchors)
run v4   yolo4.pt     608 8

# v5
for s in n s m l x; do run v5$s yolo5$s.pt 640 16; done

# v6 — standard + P6 + MBLA (12 variants)
for s in n s m l; do run v6$s yolo6$s.pt 640 16; done
for s in n s m l; do run v6${s}6 yolo6${s}6.pt 640 8; done   # P6
for s in s m l x; do run v6${s}_mbla yolo6${s}_mbla.pt 640 8; done

# v7 — base + tiny + x + P6 family
run v7      yolo7.pt       640 8
run v7tiny  yolo7-tiny.pt  640 16
run v7x     yolo7x.pt      640 8
for s in w6 e6 d6 e6e; do run v7$s yolo7-$s.pt 1280 4; done

# v8
for s in n s m l x; do run v8$s yolo8$s.pt 640 16; done

# v9 — t s m c e
for s in t s m c e; do run v9$s yolo9$s.pt 640 16; done

# v10 — n s m b l x
for s in n s m b l x; do run v10$s yolo10$s.pt 640 16; done

# v11 / v12
for s in n s m l x; do run v11$s yolo11$s.pt 640 16; done
for s in n s m l x; do run v12$s yolo12$s.pt 640 16; done

# v13 — n s l x  (no m upstream)
for s in n s l x; do run v13$s yolo13$s.pt 640 16; done

# v26
for s in n s m l x; do run v26$s yolo26$s.pt 640 16; done

echo "=== sweep done — results in $RESULTS ==="
column -ts, $RESULTS
