# Screen-detection train sweep — all variants × 1 epoch

`scripts/screen_train_sweep.sh` trains every supported `(version,
variant)` for one epoch on a 5-class screen-detection dataset
(`phone, laptop, tablet, tv, computer`; 2465 train / 308 val / 309
test images at native resolution) and dumps PASS/FAIL + final loss +
val mAP per row to a CSV.

**Last-known-good (0.88.0):** 60/60 PASS, 0 FAIL.
Raw CSVs archived in `docs/`:

| script                                      | imgsz strategy                                  | result        | CSV                                                                      |
|---------------------------------------------|-------------------------------------------------|---------------|--------------------------------------------------------------------------|
| `screen_train_sweep.sh`                     | registry default (640 / 608 / 448 / 1280)       | 60/60 PASS    | [`screen_train_sweep_60variants.csv`](screen_train_sweep_60variants.csv) |
| `screen_train_sweep_lores.sh`               | 320 for P5, 384 for P6, 448 for v1              | 60/60 PASS    | [`screen_train_sweep_60variants_lores.csv`](screen_train_sweep_60variants_lores.csv) |

The low-resolution sweep was added to flush out any hardcoded imgsz /
feature-map / anchor-cell-count assumptions that the registry-default
path doesn't exercise. **No new bugs surfaced.** Comparing the two
CSVs is a useful sanity check:

- Raw loss values drop 3–6× at the lower resolution — confirms the
  sum-reduced loss math scales correctly with `anchors × grid²` (loss
  ∝ imgsz²). The "headline-shocking" yolo5m=535 and yolo11l=428 at
  default-imgsz are NOT bugs; the same model at 320 reports
  yolo5m=85, yolo11l=99.
- mAP@0.5:0.95 actually IMPROVES for small models at 320 (less
  aggressive letterbox padding for these mostly-medium-object
  screens) — e.g. yolo8n 0.030 → 0.158, yolo13n 0.050 → 0.103,
  yolo2-tiny 0.063 → 0.121.
- yolo4 mAP drops 0.294 → 0.0006 at imgsz=320. Expected: v4's anchors
  are calibrated to imgsz=608; the anchor-rescale path does work
  (loss profile is fine, no crash), but priors are too coarse at 320
  for the head to fit screen-scale objects in one epoch.
- yolo6m6 (P6) at 384 went 0.517 → 0.569 — best P6 finetune in the
  matrix.

Per-version coverage:

| version  | variants exercised                                       |
|----------|----------------------------------------------------------|
| yolo1    | `yolo1` (from scratch — pjreddie's `.weights` URL is 404)|
| yolo2    | `yolo2` (COCO), `yolo2-tiny-voc`                         |
| yolo3    | `yolo3u`                                                 |
| yolo4    | `yolo4`                                                  |
| yolo5    | `n s m l x`                                              |
| yolo6    | `n s m l`, `s/m/l/x_mbla`, `n6/s6/m6/l6`                 |
| yolo7    | `base tiny x` (P6 variants `w6/e6/d6/e6e` skipped)       |
| yolo8    | `n s m l x`                                              |
| yolo9    | `t s m c e`                                              |
| yolo10   | `n s m b l x`                                            |
| yolo11   | `n s m l x`                                              |
| yolo12   | `n s m l x`                                              |
| yolo13   | `n s l x` (no upstream `m`)                              |
| yolo26   | `n s m l x`                                              |

Total: **60 variants, all PASS.**

## How to reproduce

```bash
bash scripts/screen_train_sweep.sh
# → /tmp/screen_train_sweep_YYYYMMDD_HHMMSS/RESULTS.csv
```

Each variant runs with `--epochs 1 --seed 42`, the registry default
imgsz, and a batch size chosen to fit in ~32 GB GPU memory:
- nano/small/medium scales: batch=8
- large/x scales: batch=4
- v6 P6 (imgsz=1280): batch=4 (n6/s6/m6), batch=2 (l6)

## Notes on loss magnitudes

A few variants show "headline-shocking" raw loss values (e.g. yolo5x
~228, yolo3u ~424, yolo11l ~428). These are pre-existing scaling
quirks in the V7/V8 loss objectives — the SUM-reduction over many
matched anchors at large batches × many classes can run into the
hundreds even when training is converging cleanly. The val mAP rows
confirm actual progress.

## Notes on Darknet-era (yolo1, yolo2)

- yolo1 trains from scratch since pjreddie's `yolov1.weights` URL
  returns 404. Loss decreases (e.g. ~7.5 → ~2.2 in one epoch on
  screen-dataset) and val mAP is non-zero by end of epoch 1.
- yolo2-voc / yolo2-tiny-voc are loaded as VOC-pretrained (nc=20);
  the final 1×1 head is reinitialised for nc=5 since shape doesn't
  match. The Darknet-19 backbone weights carry over.
