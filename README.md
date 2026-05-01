# yolocpp

Pure C++ computer-vision suite. LibTorch for training/eval, TensorRT for
deployment, OpenCV for image I/O. **No Python in the runtime path.**

**Current version: `0.17.0`** (pre-1.0).

**All 12 YOLO versions now have ONNX export wired** (v3..v13 + v26). With v6 landing in 0.17.0, every detection-supported version produces a valid ONNX file via the templated graph emitters in `src/serialization/onnx_export.cpp`.

**All 12 YOLO versions now train end-to-end.** With v10 train landing in this release, every detection-supported version (v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13/v26) trains via the templated `TrainerT<M>` runner using one of four loss classes: `V8DetectionLoss` (anchor-free DFL), `V6DetectionLoss` (VFL+SIoU+TAL), `V7DetectionLoss` (anchor-based v3-style), or `Yolo26Loss` (DFL-free NMS-free). Every code change is logged in
[CHANGELOG.md](CHANGELOG.md) with a bumped semver — see the file's
header for the pre-1.0 versioning policy. The `1.0.0` line is reserved
for whenever the project is declared stable (deliberately gated on the
maintainer's call, not a feature checklist).

A recurring **gap-audit** task (#33) sweeps the codebase periodically
for unwired pipelines, stub implementations, SKIP-gated tests, and
per-version variants not yet shipped — see CLAUDE.md
"Periodic gap-audit (recurring TODO)" for the checklist and trigger
points.

**For the full task ledger** — completed, pending, and in-flight across
the entire codebase (not just the active session) — see
**[TODO.md](TODO.md)**. It aggregates session task numbers, per-version
capability gaps, code-level TODO/FIXME comments, SKIP-gated tests, and
pre-numbered Phase work.

## Status — Phase 6 + extension tasks (val for v3..v10, v9 train) complete

| component                              | state     |
|----------------------------------------|-----------|
| Build skeleton + CMake + smoke test    | ✅ Phase 0 |
| YOLO8n architecture (Conv/C2f/SPPF/Detect/DFL) | ✅ |
| Ultralytics `.pt` weight loader (clean-room pickle parser) | ✅ |
| Letterbox + NMS + predict CLI          | ✅ |
| YOLO-format dataset + augmentation     | ✅ HSV + flip (mosaic/mixup TODO) |
| YOLO8 loss (CIoU + DFL + BCE + TAL)   | ✅ |
| Trainer (SGD + cosine LR + EMA)        | ✅ AMP TODO |
| Validator (mAP@0.5, mAP@0.5:0.95)      | ✅ |
| **ONNX export (hand-written protobuf, no Python)**       | ✅ Phase 2 |
| **TensorRT engine builder (NvOnnxParser, FP16, sm_120)** | ✅ Phase 2 |
| **TRT runtime predictor (matches libtorch detections)**  | ✅ Phase 2 |
| **Classify task (yolo8n-cls.pt → top-K)**               | ✅ Phase 3 — predict + train + val |
| **Segment task (yolo8n-seg.pt → mask overlays)**        | ✅ Phase 3 — predict + train + val (mask BCE loss, mask-mAP@0.5) |
| **Pose task (yolo8n-pose.pt → 17-keypoint skeleton)**   | ✅ Phase 3 — predict + train + val (kpt L1 + visibility BCE, OKS-mAP) |
| **OBB task (yolo8n-obb.pt → rotated boxes + NMS)**      | ✅ Phase 3 — predict + train + val (cosine angular loss, rotated-IoU mAP) |
| **Auto-resolve `model=` and `data=` from cwd / cache / Ultralytics** | ✅ Phase 3.2 |
| **Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied)** | ✅ Phase 3.2 |
| **LR-warmup formula fixed for tiny datasets**            | ✅ Phase 3.2 |
| **Trainer saves `.pt` in load_state_dict-compatible format** | ✅ Phase 3.2 |
| **All v8 scales (n / s / m / l / x) verified end-to-end** | ✅ Phase 3.3 |
| **Scale auto-detected from filename (`yolo8m.pt` → scale=m)** | ✅ Phase 3.3 |
| **Save-dir auto-increments (`runs/train` → `train2` → `train3`)** | ✅ Phase 3.4 |
| **`best.pt` saved at peak val mAP@0.5:0.95**             | ✅ Phase 3.4 |
| **Auto-attach val split when `<root>/images/val` exists** | ✅ Phase 3.4 |
| **`results.csv` per-epoch (Ultralytics-shape header)**   | ✅ Phase 3.5 |
| **`patience=N` early stopping when val mAP plateaus**    | ✅ Phase 3.5 |
| **`runs/<run>/args.yaml` reproducibility dump**          | ✅ Phase 3.6 (timestamped, Ultralytics-shape field list — 107 keys) |
| **`runs/<run>/confusion_matrix.png` rendered at end**    | ✅ Phase 3.6 |
| **`runs/<run>/{BoxPR,BoxF1,BoxP,BoxR}_curve.png`**       | ✅ Phase 3.7 |
| **`runs/<run>/labels.jpg` (per-class GT histogram)**     | ✅ Phase 3.7 |
| **`runs/<run>/results.png` (training-curve plot)**       | ✅ Phase 3.7 |
| **Multi-GPU DDP infrastructure (NCCL + all-reduce)**     | ✅ Phase 4A — compiles, world_size=1 verified; needs 2-GPU box for full validation |
| **`scripts/launch_ddp.sh <N>` torchrun-equivalent launcher** | ✅ Phase 4A |
| **`runs/<run>/train_batch{0,1,2}.jpg` augmentation sanity grids** | ✅ Phase 3.8 |
| **YOLO5 (anchorless v5u) end-to-end** — predict, **train, val** for all five scales (n/s/m/l/x) | ✅ Phase 5A + 5D |
| **YOLO3 (yolov3u: Darknet-53 + v8 DFL head, end-to-end)** | ✅ Phase 5B + 0.10/0.14 (predict / val / train / ONNX+TRT) |
| **YOLO11 — full 5 scales × 5 tasks, parity-clean forward**  | ✅ Phase 6A |
| **YOLO26 — STAL + ProgLoss train, full 5 scales × 5 tasks** | ✅ Phase 6B |
| **YOLO12 — A2C2f / AAttn detect + train + val + ONNX/TRT export** (n/s/m/l/x) | ✅ Phase 6C |
| **YOLO13 — HyperACE / FullPAD detect + train + val + ONNX/TRT export** (n/s/l/x) | ✅ Phase 6D |
| **CLI dispatch on filename pattern (`yolo3*`, `yolo5*`, `yolo8*`, …)** | ✅ Phase 5C |
| **Ultralytics-style `data=path/to/data.yaml`** (yaml-only — no directory form) | ✅ Phase 5E |
| **`data.yaml` parsed via vendored rapidyaml** (`path:` / `train:` / `val:` / `names:`) | ✅ Phase 5E |
| **Auto-download from `data.yaml`'s `download:` URL** when dataset missing | ✅ Phase 5E |
| **Model auto-inference from `.pt` state_dict shapes** — version (kernel=6 → v5, kernel=3 → v8), scale (16/32/48/64/80 → n/s/m/l/x), nc (head shape) | ✅ Phase 5E |
| **`scale=`/`version=`/`nc=` no longer required** — works on renamed `best.pt`/`last.pt` too | ✅ Phase 5E |

## YOLO version roadmap

The codebase covers **only** these twelve YOLO versions and uses the
single canonical filename / identifier convention `yolo<N>` everywhere
(`yolo3`, `yolo4`, `yolo5`, `yolo6`, `yolo7`, `yolo8`, `yolo9`, `yolo10`,
`yolo11`, `yolo12`, `yolo13`, `yolo26` — **no `v`**). When fetching legacy
upstream Ultralytics weights for v3..v10, the resolver transparently maps
the canonical name back to the `yolov<N>...pt` URL. Versions outside this
set (v1, v2, anything between v13 and v26) are intentionally not supported.

| version | year | provenance | family / changes | status |
|---------|------|------------|------------------|--------|
| **yolo3**  | 2018 | Redmon & Farhadi (official Darknet)        | Darknet-53 backbone; ships in two head forms — original anchor-based (deferred) and Ultralytics' anchor-free `yolov3u` (v8-style DFL head, used here) | ✅ **predict / val / train / ONNX+TRT export (yolov3u)** — converted on first use (fp16 → fp32). 103M params; 7 dets on `bus.jpg`, top 0.94. v3 train via `TrainerT<Yolo3>` reuses `V8DetectionLoss`. v3 ONNX (415 MB) + TRT FP32 (483 MB) match C++ predict's 7-dets baseline exactly. |
| **yolo4**  | 2020 | Bochkovskiy et al. (official Darknet)      | CSPDarknet-53 + SPP + PANet; Mish activations; v3-style anchor head | ✅ **predict / val / train / ONNX+TRT export** — Darknet `yolov4.weights` converted to our `yolo4.pt` on first use; default `imgsz=608` (anchor calibration). 6 dets on `bus.jpg`. v4 train via `V7DetectionLoss` (anchor-based with v4 scale_xy bias-fix + `exp()` wh decode). v4 ONNX (257 MB) + TRT FP32 (259 MB) match C++ baseline. |
| **yolo5**  | 2020 | Ultralytics (official)                     | CSPNet + C3 blocks, originally anchor-based; the modern `*u.pt` files use the v8 anchor-free Detect head | ✅ end-to-end (predict / train / val / ONNX + TRT export) for all 5 scales via `yolo5*u.pt` |
| **yolo6**  | 2022 | Meituan (official open-source)             | EfficientRep backbone (RepBlock for n/s, BepC3 with BottleRep for m/l) + RepBiFPANNeck + CSPSPPF (n/s) / SimSPPF (m/l) + EffiDeHead. l uses SiLU. Plus MBLA variants (s/m/l/x_mbla) and P6 variants (n6/s6/m6/l6) | ✅ **predict + val + train + ONNX/TRT export for all 12 published variants** (n/s/m/l + 4×_mbla + n6/s6/m6/l6). m/l use BepC3 + DFL; MBLA uses MBLABlock (multi-branch BottleRep3); P6 uses the 6-stage backbone + 4-level head at imgsz=1280. **Train via `V6DetectionLoss`** (VFL + SIoU + TAL); finetune mAP@0.5:0.95=0.74 on coco8. bus.jpg TRT FP32 returns: P5 n/s/m/l = 4/5/5/6, MBLA s/m/l/x = 6/6/6/5, P6 n6/s6/m6/l6 = 5/6/5/8 dets — all matching libtorch. |
| **yolo7**  | 2022 | Wang, Bochkovskiy & Liao (academic)        | ELAN backbone + ELAN-W neck + MP/DownC downsamples + SPPCSPC + (3-level IDetect for base/tiny/x; 4-level IDetect + ReOrg input for w6/e6/d6/e6e). e6e adds E-ELAN parallel ELAN sub-blocks summed via Yolo7Shortcut | ✅ **predict + val for all 7 variants**, **train + ONNX+TRT export** for the IDetect anchor-decode form. v7 train via `V7DetectionLoss` (scale_xy=2.0 + `(sigmoid*2)²` wh decode); base finetune mAP@0.5:0.95=0.72 on coco8. v7 ONNX walks the per-scale yaml via the public `yolo7_yaml_for(scale)` accessor. |
| **yolo8**  | 2023 | Ultralytics (official)                     | CSP + C2f backbone, anchor-free DFL Detect, TAL assigner | ✅ **full** — train / val / predict / export across 5 scales × 5 tasks (detect / segment / classify / pose / OBB) |
| **yolo9**  | 2024 | Wang, Yeh & Liao (academic)                | GELAN backbone (RepNCSPELAN4 + ADown/AConv + SPPELAN + ELAN1) + v8-style anchor-free Detect head; PGI auxiliary branch dropped at deploy. e adds CBLinear/CBFuse two-pass backbone | ✅ **predict + val + train + ONNX/TRT export for all 5 scales (t / s / m / c / e)**. e ONNX (added 0.20.0) emits the 43-layer two-pass graph: a primary backbone with 5 CBLinear taps, a secondary backbone that re-ingests the input image and pulls CBLinear branches via CBFuse (`Slice` + `Resize(mode=nearest)` + `Add`) at each downsample, plus the standard GELAN head. v9{c,e} TRT FP32 returns 5 dets on bus.jpg matching libtorch. PGI auxiliary branch is intentionally not wired (training-only upstream). |
| **yolo10** | 2024 | Tsinghua MIG (academic, Ultralytics-hosted)| SCDown + C2f + C2fCIB + SPPF + PSA backbone; v10Detect (one2one head used at deploy → effectively NMS-free) | ✅ **predict + val + train + ONNX+TRT export for all 6 scales (n / s / m / b / l / x)**. Single-head training uses the deploy one2one head with `V8DetectionLoss`; paper §3.1 dual-head consistent assignment (added 0.22.0) trains a parallel one2many head (legacy=true cv3) with `V10DualLoss` = `V8DetectionLoss(o2m, topk=10)` + `V8DetectionLoss(o2o, topk=1)` — enable via `dual_head=true`. TRT FP32 disables TF32 per-version (the RepVGGDW 7×7 dwconv stack accumulates enough TF32 mantissa loss to saturate cls); after the fix every scale matches ORT (5 dets on bus.jpg, top conf 0.94–0.97). |
| **yolo11** | 2024 | **Ultralytics (official)**                 | Refined CSP: C3k2 (kernel-tunable C3) + C2PSA (position-sensitive attention); v11 Detect head with depthwise-separable cv3 (DWConv→Conv) | ✅ **full** — train / val / predict / ONNX + TRT export across 5 scales × 5 tasks. Forward bit-exact vs Ultralytics Python through layer 22 (parity harness verified). Full-COCO val mAP@0.5:0.95 within 0.05% of Ultralytics' own `m.val(rect=False)` on n/s. |
| **yolo12** | 2025 | Tian et al., Ultralytics-hosted            | Attention-centric: A2C2f (Area-Attention C2f) with windowed global attention, gamma-gated outer residual at l/x | ✅ **detect end-to-end** — train / val / predict / ONNX + TRT export across all 5 scales (n/s/m/l/x). Forward parity-clean (5/5/5/6/5 detections matching Python on bus.jpg). ONNX max\|Δ\| ≤ 1.8e-7 vs Python through onnxruntime. Task heads (segment / pose / obb / classify) ⏳ **planned future session** — Ultralytics ships only detect weights for v12, so we'll train our own task heads on COCO. |
| **yolo13** | 2025 | Lei et al. (iMoonLab fork)                 | HyperACE (hypergraph adaptive correlation enhancement) + FullPAD distribution + DSConv depthwise-separable variants + V13AAttn (separate qk/v convs, k=5 pe) | ✅ **detect end-to-end** — train / val / predict / ONNX + TRT export across n/s/l/x (iMoonLab does not ship `m`). Forward cls-channel max\|Δ\| ≤ 7.6e-10 vs iMoonLab Python on all 4 scales. ONNX max\|Δ\| ≤ 1.8e-7. Task heads ⏳ **planned future session** — iMoonLab ships only detect weights, so we'll train our own task heads on COCO. |
| **yolo26** | 2025 | **Ultralytics (official, preview)**        | DFL-free Detect head, end-to-end NMS-free inference, ProgLoss + STAL assigner — edge/mobile-first | ✅ **full** — train / val / predict / ONNX + TRT export across 5 scales × 5 tasks |
| RT-DETR    | 2023 | Baidu (official)                           | HGNetv2 + AIFI + deformable-attention decoder; transformer-based, NMS-free | 🟡 architecture probed (Phase 4 — transformers) |

Stub status (🟡) means the header / source files exist with the right
class name and namespace, the build links cleanly, and `mode=predict`
will produce a clear `not implemented yet` runtime error pointing at the
header that explains the design plan.

Pending status (⏳) means the architecture is end-to-end for detect, but
task variants (segment/pose/obb/classify) are not yet trained because
neither upstream publishes those weights. Planned to train our own on
COCO in a future session.

### Capability matrix at a glance (detect)

```
              arch     predict       val      train             ONNX/TRT export
yolo3         ✅       ✅(u form)    ✅       ✅(u form)        ✅
yolo4         ✅       ✅            ✅       ✅                ✅
yolo5         ✅       ✅            ✅       ✅                ✅
yolo6         ✅       ✅(all 12)    ✅       ✅                ✅(all 12)
yolo7         ✅       ✅(all 7)     ✅       ✅(base)          ✅
yolo8         ✅       ✅            ✅       ✅                ✅
yolo9         ✅       ✅(t,s,m,c,e) ✅       ✅                ✅(t,s,m,c,e)
yolo10        ✅       ✅(all 6)     ✅       ✅(single + dual) ✅(noTF32)
yolo11        ✅       ✅            ✅       ✅                ✅
yolo12        ✅       ✅            ✅       ✅                ✅
yolo13        ✅       ✅            ✅       ✅                ✅
yolo26        ✅       ✅            ✅       ✅                ✅
```

Outstanding work, by version (each is its own multi-session task):

- **v3 train** — `Yolo3Impl` needs `forward_train` + `(scale, nc)` ctor
  reorder; can reuse `V8DetectionLoss` since yolov3u uses v8's
  anchor-free DFL Detect head. Tracked as #29.
- **v4/v7 train** — anchor-based v3-style loss class (BCE-obj +
  BCE-cls + IoU-box, multi-anchor assignment). Tracked as #30.
- **v6 train** — VFL + SIoU + TAL loss (anchor-free, with the
  knowledge-distillation `reg_preds_dist` branch as the DFL target).
  Tracked as #31.
- **v10 train** — dual-head consistent assignment (one2many +
  one2one). Needs arch rework to keep `one2many` in `Yolo10Impl`
  (currently dropped at conversion via `convert_yolov10_pt`). Tracked
  as #32.
- **v3/v4/v6/v7/v9/v10 ONNX + TRT export** — anchor-decode emitter
  for v4/v7; dual-branch decode (or just `reg_preds`) for v6; v3u/v9
  reuse the existing v8 `emit_detect_v8` once the trainer's
  `forward_train` plumbs through; v10 needs the NMS-free postprocess
  (per-class topk on the one2one head). Tracked as #19.
- **v6 high-res P6 variants** (`yolov6{n,s,m,l}6.pt`) and **MBLA
  variants** (`yolov6x_mbla.pt`, etc.) — separate tasks #23 / #24.
- **v10 s/m/b/l/x predict** — different channel widths and
  per-layer dispatch (which layers are SCDown vs Conv vs C2fCIB vs
  C2f differs per scale). Tracked as #21.

| dependency | version            |
|-----------|--------------------|
| LibTorch  | 2.11.0+cu130       |
| TensorRT  | 10.14.1.48+cuda13.0|
| CUDA tk   | 13.0.88            |
| OpenCV    | 4.6.0              |

## Build

```bash
./scripts/install_third_party.sh           # ~5 GB download, idempotent
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The test suite covers every layer of the stack:

| test                 | what it checks                                       |
|----------------------|------------------------------------------------------|
| `smoke_test`         | LibTorch CUDA + custom CUDA kernel + OpenCV + TRT    |
| `test_yolo8_arch`    | YOLO8n architecture: param count (~3.16M), strides, output shapes |
| `test_pt_loader`     | Loads real `yolo8n.pt`, verifies all 355 keys/shapes |
| `test_predict`       | End-to-end inference on `bus.jpg`, ≥1 person + ≥1 bus |
| `test_train_overfit` | Tiny synthetic dataset, training loss decreases ≥ 2× |
| `test_v{3,4,6,7,9,10}_e2e` | Per-version predict end-to-end, gated on cache `.pt` availability |
| `test_val_v3_v10`    | Templated `engine::validate` runs on v3/v4/v6/v7/v9/v10 holders |
| `test_v9_train`      | yolo9c finetune-on-coco8 smoke (loss decreases, checkpoint written) |
| `test_v13_full`      | v13 forward parity vs iMoonLab Python (cls max\|Δ\| ≤ 7.6e-10) |
| `test_v13_ada`       | v13 hypergraph modules (AdaHyperedgeGen / HyperACE) bit-exact |

## CLI

Two argument styles, both supported:

```
# Ultralytics-style (canonical)
yolocpp task=detect mode=train   model=yolo8n.pt data=path/to/data.yaml epochs=100
yolocpp task=detect mode=val     model=yolo8n.pt data=path/to/data.yaml
yolocpp task=detect mode=predict model=yolo8n.pt source=bus.jpg
yolocpp task=detect mode=export  model=yolo8n.pt format=trt fp16=true
yolocpp mode=benchmark           model=yolo8n.pt source=bus.jpg

# Legacy subcommand-style
yolocpp predict   --weights=yolo8n.pt  --source=bus.jpg --out=out.jpg
yolocpp predict   --weights=yolo8n.trt --source=bus.jpg --out=out.jpg
yolocpp val       --weights=yolo8n.pt  --data=<root>
yolocpp train     --data=<root> [--epochs=100] [--scale=n] [--weights=yolo8n.pt]
yolocpp export    --format=onnx --weights=yolo8n.pt --out=yolo8n.onnx
yolocpp export    --format=trt  --weights=yolo8n.pt --out=yolo8n.trt --fp16
yolocpp benchmark --weights=yolo8n.pt  --source=bus.jpg
yolocpp info
```

`data=` accepts **only a `.yaml`/`.yml` file** in the kv-style form. The yaml's
`path:` / `train:` / `val:` / `names:` / `download:` are honored — if the
dataset isn't on disk, the `download:` URL is fetched and unzipped automatically.

`model=` is enough on its own — version (v5/v8), scale (n/s/m/l/x) and `nc`
are auto-inferred from the `.pt`'s actual layer shapes (`model.0.conv.weight`
kernel + first dim, head's `cv3.0.2.weight` first dim). This works for renamed
checkpoints (`best.pt`, `last.pt`) where the filename carries no version
letter. Pass `version=` / `scale=` / `nc=` only to override.

All five `task` values now support train + val + predict + export (detect)
and train + val + predict (classify/segment/pose/obb):

```
yolocpp task=classify mode=train   data=DATA model=yolo8n-cls.pt epochs=30
yolocpp task=segment  mode=train   data=DATA model=yolo8n-seg.pt epochs=30
yolocpp task=pose     mode=train   data=DATA model=yolo8n-pose.pt epochs=30
yolocpp task=obb      mode=train   data=DATA model=yolo8n-obb.pt epochs=30
yolocpp task=segment  mode=val     data=DATA model=runs/segment/last.pt
```

Dataset layouts:
- **classify**: `<root>/{train,val}/<class_name>/img.jpg`
- **segment**: same as detect, with optional polygon vertices appended
  to each label line (`cls cx cy w h x1 y1 x2 y2 ... xN yN`)
- **pose**: same as detect, with `K` triplets `(kx, ky, v)` appended
  per line (visibility v ∈ {0, 1, 2})
- **obb**: same as detect, with an angle (radians) appended:
  `cls cx cy w h angle`

Loss & metric specifics (minimum-viable; production parity is Phase 3.2):
- **classify**: cross-entropy ; top-1 / top-5
- **segment**: per-positive-anchor mask BCE on `sigmoid(coefs × protos)`
  vs polygon-rasterized mask ; mask mAP@0.5
- **pose**: L1 on `(x, y)` weighted by visibility + visibility BCE ; OKS-mAP@0.5
- **obb**: `1 - cos(pred_angle - gt_angle)` periodic-friendly loss on
  the closest anchor ; rotated-IoU mAP@0.5

### Benchmark output (RTX 5090, FP16 — bus.jpg, 200 iters)

```
  backend                  median (ms)   p95 (ms)   mean (ms)   img/s    dets
  ──────────────────────  ────────────  ─────────  ──────────  ───────  ─────
  PT (libtorch FP32)              3.90       4.38        3.94    256.4      6
  TRT FP32                        2.04       2.20        2.05    491.3      6
  TRT FP16                        1.61       1.69        1.61    622.6      6

  Speedup vs PT:
    TRT FP32                1.92x
    TRT FP16                2.43x
```

Dataset layout:
```
<root>/images/{train,val}/<id>.jpg
<root>/labels/{train,val}/<id>.txt   # one line: cls cx cy w h, all in [0,1]
```

## Architecture map

```
yolocpp/
  models/yolo8         Conv → C2f → SPPF → Detect (DFL)
                        scale variants n/s/m/l/x
  serialization/
    pt_loader           Reads Ultralytics .pt zip + clean-room pickle parser
                        → state_dict {dotted_name → torch::Tensor}
  inference/
    letterbox           letterbox + image_to_tensor + scale_boxes
    nms                 class-aware NMS (CPU)
    predictor           load weights → preprocess → forward → NMS → unscale
  datasets/
    yolo_dataset        YOLO-format (.txt) loader, HSV + flip augmentation
  losses/
    yolo8_loss         TAL assigner + CIoU + DFL + BCE
  engine/
    trainer             SGD/momentum/Nesterov + warmup + cosine LR + EMA
    validator           full pass over val set → mAP via metrics/map
  metrics/map           per-class AP at IoU={0.5, 0.5..0.95}
```

## Architecture decisions worth knowing

- **`.pt` loader is a clean-room pickle parser**, not libtorch's
  `torch::jit::Unpickler`. Reason: the standard Unpickler trips on the
  BUILD opcode for arbitrary Python classes (`DetectionModel` etc.). Our
  parser handles every opcode produced by `torch.save` and treats unknown
  GLOBALs as opaque object stubs while still extracting the underlying
  tensor data.
- **Modules are constructed in the same order as Ultralytics' yaml**, so
  `named_parameters()` iteration order matches the checkpoint exactly.
- **`-Wl,--disable-new-dtags`** so `libnvinfer.so`'s `dlopen` of its
  sm_120 resource library finds it via the executable's rpath. See CLAUDE.md.

## Phase 2 export — how it works

- **`serialization/onnx_export.cpp`** writes the ONNX protobuf wire format
  by hand. We don't link libprotobuf and don't run any Python tracer. The
  exporter walks our `Yolo8Detect` ModuleList, emits one ONNX node per
  layer (Conv, BN folded into Conv, SiLU = Sigmoid+Mul, Add, MaxPool,
  Concat, Resize/Upsample, Slice, Softmax, Sub, Mul), and bakes the
  Detect head's DFL projection + anchor/stride decoder into the graph.
  Output is `[N, 4 + nc, A]` in input pixels with sigmoided class scores
  — same shape libtorch's `forward_eval` returns.
- **`serialization/trt_export.cpp`** uses `nvonnxparser::IParser` to read
  our ONNX, configures a single optimization profile (default
  batch=1, imgsz=640), enables FP16 on Blackwell, and saves the
  serialized plan. `~6 seconds at builder_opt_level=1`, ~30 seconds at
  default level.
- **`inference/trt_predictor.cpp`** loads the plan, allocates input/output
  GPU buffers once, runs `enqueueV3`, copies output back, and reuses our
  CPU NMS + `scale_boxes`. Returns the same `Detection` struct the
  libtorch `Predictor` does.

`test_trt_export` builds the engine on every run and verifies the TRT
output matches libtorch detections within 30 px box-center tolerance and
0.20 confidence tolerance — in practice they agree to FP16 rounding.

## Documentation map

- **[CHANGELOG.md](CHANGELOG.md)** — chronological changelog covering
  every phase, version, and parity gotcha. **Start here when you want
  to know what changed and why.**
- **[CLAUDE.md](CLAUDE.md)** — internal developer notes / Claude Code
  briefing. Detailed architecture commitments, per-version parity
  notes (BN eps, CSPSPPF cat order, RepVGG fusion gotchas, etc.),
  build / toolchain rationale (why DT_RPATH, why cu130, etc.).
- **README.md** (this file) — public-facing entry point: status,
  build/run, CLI surface, supported versions.

## What's deliberately deferred

- **v3/v4/v6/v7/v10 train + v3..v10 ONNX/TRT export** — see the per-version
  outstanding-work list above. Each is a multi-session deliverable:
  v4/v7 share an anchor-based v3-style loss class, v6 needs VFL +
  SIoU + TAL, v10 needs dual-head consistent assignment + arch rework
  to keep one2many. v3 / v9 / v10 export piggyback on v8's emitter
  once their `forward_train` (or one2one-only graph for v10) is
  exposed.
- **v12 / v13 task heads (segment / pose / obb / classify)** — neither
  Ultralytics nor iMoonLab publishes task weights upstream (only detect
  ships). **Planned future session:** train our own task heads on COCO
  using the existing templated `Trainer` (already supports v12/v13
  detect) — v12 = 5 scales × 4 tasks = 20 runs, v13 = 4 scales × 4 tasks
  = 16 runs. Yolo12 task scaffolding already exists in
  `src/models/yolo12_tasks.cpp` but is untested against real weights;
  Yolo13 task module declarations are not yet written. See CLAUDE.md
  "Task variants for v12 / v13 — not available upstream".
- **Mosaic / mixup augmentation**: full multi-image augmentation in C++
  is straightforward but ~600 lines we haven't needed yet.
- **AMP (mixed-precision training)**: Trainer is FP32-only. Adding
  `torch::autocast` is a future change.
- **Multi-threaded data prefetch**: dataset is synchronous. With OpenCV
  decode + CUDA inference, the IO bottleneck on a 5090 is real but
  fixable later.
- **TRT INT8 calibration** and dynamic-shape multi-batch profiles: easy
  to add on top of `TrtBuildConfig` once a calibration set exists.
