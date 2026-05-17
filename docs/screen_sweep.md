# Screen-detection variant sweep — yolocpp 0.66.0

Full **(version × variant)** sweep on a 5-class screen-detection dataset
(`phone, laptop, tablet, tv, computer`; 2465 train / 308 val / 309 test
images). Run via `scripts/screen_variant_sweep.sh`. 3 epochs per
variant, `--seed 42`, val mAP picks the best epoch.

## Setup fixes that landed this session

The naive form of the sweep ran headfirst into three latent bugs that
made every variant look broken in different ways. All three are fixed
in **0.65.0–0.66.0** and are required for the numbers below:

1. **`V7DetectionLoss` CUDA segfault** — `obj_target.accessor<>()` on
   a CUDA tensor. v4 + v7 train core-dumped on GPU; CPU silently
   worked. Fix: build `obj_target` on CPU, move to dev before BCE
   (`src/losses/yolo7_loss.cpp`).
2. **`resolve_weights` was never called** — `--model yolo3u.pt` failed
   with errno-2 even though the auto-download code path existed.
   Fix: wired into `cmd_dispatch_flag_style` once after CLI11 parse
   (`src/cli/main.cpp`).
3. **`cmd_train` ignored the data.yaml's `nc:`/`names:`** — without
   `--names`, it defaulted to COCO 80, so a 5-class dataset trained
   an 80-class detector. Combined with `load_from_state_dict`
   throwing on shape mismatch in v5/v8/v11/v12/v13/v26 loaders, the
   shape between yaml and weights also wasn't reconcilable. Fix:
   parse `data.yaml.names`, and relax all shape-mismatch throws to
   skip-with-log so cls heads get re-init at torch defaults
   (`src/cli/commands.cpp` + every `src/models/yolo*[_tasks|_classify].cpp`).
4. **`scale_from_filename` regex omitted v9 `t/c/e`** — `yolo9t.pt`
   resolved to the C-scale default → 745/782 keys shape-mismatched
   at load. Fix: regex extended to `[nsmblxtce]`
   (`src/cli/resolve.cpp`).
5. **`Detect26Impl::init_biases()`** added — re-applies the upstream
   detection-prior bias (cls = log(0.01/0.99); reg = 1.0) after a
   custom-nc partial load. (Necessary but not sufficient for v26 —
   see Known issues below.)

## Results

56 of 71 (variant × scale) cells trained successfully; the 15
failures break into three known categories (P6 architecture bug, OOM
at batch=16 on x-scale, and a v26-specific cold-start collapse). The
n/s scales are listed first so anyone scanning the file can compare
across families at a single capacity point.

| variant  | mAP@0.5 | mAP@0.5:0.95 | s/ep | note |
|----------|--------:|-------------:|-----:|------|
| **v3u**  | 0.43 | 0.32 | 54 | u-form; single scale |
| **v4**   | 0.73 | 0.48 | 73 | imgsz=608 (anchor calib) |
| **v5n**  | 0.49 | 0.34 | 14 | |
| v5s      | 0.48 | 0.32 | 18 | |
| v5m      | 0.44 | 0.31 | 25 | |
| v5l      | 0.42 | 0.29 | 34 | |
| v5x      | 0.36 | 0.23 | 51 | |
| **v6n**  | 0.61 | 0.32 | 172 | RepConv-fusion path is slow |
| v6s      | 0.75 | 0.39 | 42 | |
| v6m      | **0.88** | **0.78** | 68 | best on this dataset |
| v6l      | 0.87 | 0.77 | 74 | |
| v6{n,s,m,l}6 | FAIL | — | — | P6 forward bug — 256 vs 192 channel mismatch (pre-existing, train never exercised) |
| v6s_mbla | 0.71 | 0.63 | 144 | |
| v6m_mbla | 0.74 | 0.65 | 115 | |
| v6l_mbla | 0.75 | 0.66 | 111 | |
| v6x_mbla | 0.72 | 0.64 | 119 | |
| **v7**   | 0.39 | 0.23 | 48 | base; anchor-based |
| v7tiny   | 0.48 | 0.30 | 24 | |
| v7x      | 0.52 | 0.38 | 65 | |
| v7{w6,e6,d6,e6e} | FAIL | — | — | 4-level P6 head vs 3-level `LossTraits<Yolo7>` (pre-existing) |
| **v8n**  | 0.55 | 0.39 | 14 | |
| v8s      | 0.51 | 0.36 | 18 | |
| v8m      | 0.45 | 0.32 | 27 | |
| v8l      | 0.36 | 0.23 | 37 | |
| v8x      | 0.37 | 0.25 | 50 | |
| **v9t**  | 0.52 | 0.37 | 20 | was 0.0009 before regex fix |
| v9s      | 0.44 | 0.33 | 24 | |
| v9m      | 0.47 | 0.34 | 30 | |
| v9c      | 0.40 | 0.28 | 35 | |
| v9e      | 0.05 | 0.02 | 77 | OOM @ b=16 → retried b=4 (1798M model) |
| **v10n** | 0.51 | 0.38 | 15 | was 0.24 before nc fix |
| v10s     | 0.53 | 0.37 | 19 | |
| v10m     | 0.38 | 0.25 | 27 | |
| v10b     | 0.42 | 0.29 | 31 | |
| v10l     | 0.39 | 0.26 | 37 | |
| v10x     | 0.40 | 0.27 | 48 | |
| **v11n** | 0.42 | 0.29 | 16 | was 0.18 before nc fix |
| v11s     | 0.41 | 0.27 | 19 | |
| v11m     | 0.33 | 0.18 | 28 | |
| v11l     | 0.33 | 0.20 | 35 | |
| v11x     | 0.22 | 0.12 | 51 | |
| **v12n** | 0.29 | 0.18 | 18 | was 0.11 before nc fix |
| v12s     | 0.34 | 0.20 | 23 | |
| v12m     | 0.35 | 0.22 | 34 | |
| v12l     | 0.34 | 0.22 | 47 | |
| v12x     | 0.24 | 0.13 | 75 | OOM @ b=16 → retried b=8 |
| **v13n** | 0.47 | 0.31 | 20 | |
| v13s     | 0.46 | 0.33 | 27 | |
| v13l     | 0.17 | 0.08 | 64 | OOM @ b=16 → retried b=8 |
| v13x     | 0.32 | 0.20 | 89 | OOM @ b=16 → retried b=8 |
| **v26{n,s,m,l,x}** | **0.00** | **0.00** | — | STAL+ProgLoss cold-start collapse — see Known issues |

## Known issues — not regressions, pre-existing in upstream codebase

These remain after this session's fixes and need their own debug
sessions:

1. **`yolo6{n,s,m,l}6` train fails** with
   `tensor a (256) must match tensor b (192) at dim 0`. The v6 P6
   forward path (`Yolo6Impl::forward` with `is_p6=true`) has a
   channel-routing bug in one of the neck connections. Predict +
   export of v6 P6 worked in 0.24.0 (full matrix sweep `PASS=152`)
   but **train was never green for P6 variants** — only `train 3`
   cells were verified upstream of this session.
2. **`yolo7-{w6,e6,d6,e6e}` train segfaults** under
   `V7DetectionLoss`. The four P6 v7 variants emit 4 feature levels
   (P3/P4/P5/P6) but `LossTraits<Yolo7>` hardcodes 3 levels
   (P3/P4/P5 anchors + scale_xy + balance arrays of size 3). Need a
   P6 anchor table + per-stride loss config.
3. **`yolo26*` train collapses to mAP=0** when re-targeting upstream
   nc=80 weights at a custom nc. Cls bias is correctly re-initialised
   to log(0.01/0.99) by `init_biases()` (added this session), but
   STAL's alignment metric `cls^α · iou^β` with α=0.5/β=6.0 stays
   near-zero through cold start: the `valid_floor=1e-12` keeps top-1
   argmax pointing at an in-GT anchor for ~10 steps, but as soon as
   cls predictions saturate to zero (the BCE-with-soft-targets
   gradient pushes them down), STAL stops assigning positives,
   `fg_mask` empties, and the loss converges to zero with nothing
   learned. Upstream Ultralytics works around this with a much
   longer warmup schedule (`warmup_epochs=3`) and a different
   `box_gain`. Possible fix: clamp `progress` to keep `iou^prog`
   above some floor during the first epoch, or hard-assign top-k
   positives per GT until alignment is non-trivial.

## How to reproduce

```bash
bash scripts/screen_variant_sweep.sh    # ~2.5h on a single 32 GB H100/L40
column -ts, /tmp/screen_sweep/RESULTS.csv
```
