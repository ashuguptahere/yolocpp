# TODO — yolocpp full task ledger

**Single source of truth for everything done, in flight, and pending across the entire codebase** — not just the active session. Compiled by walking session task #1..#33, README.md "deferred" sections, CLAUDE.md "Remaining gaps", per-file `TODO`/`FIXME`/`deferred` comments, SKIP-gated tests, and the git history (Phases 0..6 pre-dating session task numbering).

This file is maintained as part of recurring task **#33** (gap-audit) — see CLAUDE.md `## Periodic gap-audit (recurring TODO)` for the audit checklist and trigger points. Last refreshed at version `0.24.0` (2026-05-01).

> **Snapshot at 0.24.0**: every numbered task #12..#45 closed except recurring #33. Full matrix sweep (`scripts/full_matrix_sweep.sh`) reports `PASS=152 FAIL=0 SKIP=0`. ctest 31/31 green. See `SESSION_DIGEST.md` for the per-version landing map across this session's work.

Legend: ✅ done · 🟡 partial / scaffolded · ⏳ planned · ❌ not started · 🔁 recurring

---

## 1. Completed work

### 1.1 Build & toolchain (Phase 0)
- ✅ CMake + LibTorch (cu130) + TensorRT (10.14.1.48 / cuda13.0) + OpenCV 4.6.0 wired up.
- ✅ `scripts/install_third_party.sh` idempotent, ~5 GB download.
- ✅ `-Wl,--disable-new-dtags` so `libnvinfer.so`'s `dlopen` of sm_120 resource libraries finds them via the executable's rpath.
- ✅ Smoke test exercises custom CUDA kernel + libtorch CUDA + OpenCV + TRT.

### 1.2 yolo8 — full pipeline (Phase 1)
- ✅ Architecture (Conv / C2f / SPPF / Detect with DFL).
- ✅ Clean-room pickle parser for Ultralytics `.pt` (handles every opcode `torch.save` produces; treats unknown GLOBAL classes as opaque object stubs while still extracting tensor data).
- ✅ Letterbox + NMS + predict CLI.
- ✅ YOLO-format dataset loader, HSV + flip augmentation.
- ✅ Yolo8 loss (TAL assigner + CIoU + DFL + BCE).
- ✅ Trainer: SGD/momentum/Nesterov + warmup + cosine LR + EMA.
- ✅ Validator: full pass over val set → mAP via `metrics/map`.
- ✅ All 5 scales (n / s / m / l / x) verified end-to-end (Phase 3.3).

### 1.3 Export (Phase 2)
- ✅ Hand-written ONNX protobuf emitter (no libprotobuf, no Python tracer). Emits one ONNX node per layer; bakes DFL projection + anchor/stride decoder.
- ✅ TensorRT engine builder via `nvonnxparser::IParser`. Single optimization profile, FP16 on Blackwell, ~6 s at builder_opt_level=1.
- ✅ TRT runtime predictor matches libtorch detections within 30 px / 0.20 conf.

### 1.4 Task heads on v8 / v11 / v26 (Phase 3 + 6A + 6B)
- ✅ classify — predict + train + val.
- ✅ segment — predict + train + val (mask BCE loss, mask-mAP@0.5).
- ✅ pose — predict + train + val (kpt L1 + visibility BCE, OKS-mAP).
- ✅ obb — predict + train + val (cosine angular loss, rotated-IoU mAP).
- ✅ All 75 (version × task × scale) combinations for v8/v11/v26 load Ultralytics' shipped weights and produce non-empty output on bus.jpg.
- ✅ Task ONNX export — graph emitters for segment proto / pose kpt decode / OBB dist2rbox / classify head. ONNX max\|Δ\| ≤ 0.004 across tasks.

### 1.5 Production-training extras (Phases 3.2 – 3.8)
- ✅ Auto-resolve `model=` and `data=` from cwd / cache / Ultralytics URL.
- ✅ Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied).
- ✅ LR-warmup formula fixed for tiny datasets.
- ✅ Trainer saves `.pt` in `load_state_dict`-compatible format.
- ✅ Scale auto-detected from filename.
- ✅ Save-dir auto-increments (`runs/train` → `train2` → `train3`).
- ✅ `best.pt` saved at peak val mAP@0.5:0.95.
- ✅ Auto-attach val split when `<root>/images/val` exists.
- ✅ `results.csv` per-epoch (Ultralytics-shape header).
- ✅ `patience=N` early stopping when val mAP plateaus.
- ✅ `runs/<run>/args.yaml` reproducibility dump (timestamped, 107-key Ultralytics-shape).
- ✅ `runs/<run>/confusion_matrix.png` rendered at end.
- ✅ `runs/<run>/{BoxPR,BoxF1,BoxP,BoxR}_curve.png`.
- ✅ `runs/<run>/labels.jpg` per-class GT histogram.
- ✅ `runs/<run>/results.png` training-curve plot.
- ✅ `runs/<run>/train_batch{0,1,2}.jpg` augmentation sanity grids.

### 1.6 Multi-GPU (Phase 4A)
- 🟡 NCCL + all-reduce wired into `TrainerT`. Compiles, world_size=1 verified end-to-end. **Two-GPU box validation pending hardware.**
- ✅ `scripts/launch_ddp.sh <N>` torchrun-equivalent launcher.

### 1.7 Auto-resolve / data.yaml (Phase 5C + 5E)
- ✅ CLI dispatch on filename pattern (`yolo3*`, `yolo5*`, `yolo8*`, …).
- ✅ Ultralytics-style `data=path/to/data.yaml` (yaml-only — no directory form).
- ✅ `data.yaml` parsed via vendored rapidyaml (`path:` / `train:` / `val:` / `names:` / `download:`).
- ✅ Auto-download from `data.yaml`'s `download:` URL when dataset missing.
- ✅ Model auto-inference from `.pt` state_dict shapes — version, scale, nc.
- ✅ `scale=` / `version=` / `nc=` no longer required (works on renamed `best.pt` / `last.pt`).

### 1.8 yolo3 (Phase 5B + task #13 + #29)
- ✅ Predict end-to-end (Ultralytics' `yolov3u`) — Darknet-53 + v8 anchor-free DFL head. 103M params; 7 dets on bus.jpg.
- ✅ Val (task #17).
- ✅ **Train** (task #29) — TrainerT<Yolo3> via default V8DetectionLoss; `forward_train` returns per-scale raw features. Smoke: yolov3u finetune on coco8, loss 7.87 → 6.63 in 2 epochs at lr=1e-3, mAP@0.5:0.95 0.723 → 0.758.
- ✅ Converter `convert_yolov3_pt` (fp16 → fp32 + drop `num_batches_tracked`).

### 1.9 yolo4 (task batch)
- ✅ Predict end-to-end — CSPDarknet-53 + Mish + SPP + PANet + v3-style anchor head, 64.36M params, default `imgsz=608`.
- ✅ Val (task #17).
- ✅ Darknet `yolov4.weights` → `yolo4.pt` converter (CFG-DFS-order tensor walk, AlexeyAB scale_x_y bias-fix, obj*cls fusion).

### 1.10 yolo5 (Phase 5A + 5D)
- ✅ Predict / train / val for all 5 scales via `yolo5*u.pt` (anchorless v5u).
- ✅ ONNX + TRT export.

### 1.11 yolo6 (task batch + #14 + #22 + #23)
- ✅ Predict end-to-end for all 4 standard scales (n / s / m / l).
- ✅ Val (task #17).
- ✅ Converter `convert_yolov6_pt` (RepVGG fusion + key rename + fp16 → fp32; lookahead-restricted `.block.` strip).
- ✅ Three structural parity gotchas resolved: BN eps=1e-5 on RepVGG branches, CSPSPPF cat order, `reg_preds` (not `reg_preds_dist`) at eval.
- ✅ `BepC3` + `BottleRep` + `SimSPPF` for m/l (#22 + #14).
- ✅ `V6ActScope` SiLU toggle for v6l (`training_mode='conv_silu'` upstream).
- ✅ MBLA variants (#23) — `MBLABlock` + `BottleRep3` modules; `Yolo6Variant`-aware Backbone/Neck dispatch; auto-resolve for `yolov6{s,m,l,x}_mbla.pt`. All 4 scales predict + val on coco8.

### 1.12 yolo7 (tasks #15 + #25 + #26 + #27 + #28)
- ✅ Predict end-to-end for ALL 7 published variants — base / tiny / x / w6 / e6 / d6 / e6e.
- ✅ Val (task #17, P6 variants default `imgsz=1280`).
- ✅ Converter `convert_yolov7_pt` (RepConv fusion + fp16 → fp32).
- ✅ `V7ActScope` LeakyReLU toggle for tiny.
- ✅ `ReOrg` + 4-level IDetect for w6 / e6 / d6 / e6e.
- ✅ `DownC` + 6-inner ELAN for e6.
- ✅ 8-inner ELAN + 10-element cat for d6.
- ✅ E-ELAN + `Yolo7Shortcut` for e6e (helper-driven 262-entry yaml via `e6e_bb_stage` / `e6e_head_td_stage` / `e6e_head_bu_stage`).

### 1.13 yolo9 (task batch + #16 + #18)
- ✅ Predict end-to-end for all 5 scales (t / s / m / c / e).
- ✅ Val (task #17).
- ✅ **Train** (task #18) via `TrainerT<Yolo9>` reusing `V8DetectionLoss`. Smoke: yolo9c finetune on coco8, loss 12.4 → 7.3 in 2 epochs at lr=1e-3.
- ✅ Converter `convert_yolov9_pt` (RepConv fusion + fp16 → fp32).
- ✅ `Yolo9Impl::forward_train` exposing per-scale raw `[B, 4*reg_max+nc, H_i, W_i]` features.
- ✅ `make_divisible(args[0], 8)` round-up gotcha for v9m caught.
- ✅ `CBLinear` / `CBFuse` two-pass backbone for v9e (#16).

### 1.14 yolo10 (task #12 + #21)
- ✅ Predict end-to-end for **all 6 scales (n / s / m / b / l / x)**.
- ✅ Val (task #17).
- ✅ Converter `convert_yolov10_pt` (drops one2many head, renames one2one_cv2/3 → cv2/3, fuses RepVGGDW pairs, fp16 → fp32).
- ✅ Modules: `SCDown`, `RepVGGDW`, `CIB` (with `e` parameter), `C2fCIB`, `PSA`. Caught the upstream `e=1.0` override on inner CIBs.

### 1.15 yolo11 (Phase 6A)
- ✅ Full 5 scales × 5 tasks. Parity-clean forward (bit-exact vs Python through layer 22 across all 5 scales × {detect, classify, segment, pose, obb}).
- ✅ Full-COCO val mAP@0.5:0.95 within 0.05% of Ultralytics' `m.val(rect=False)` on n/s.
- ✅ ONNX + TRT export.
- ✅ New shared infrastructure: `DWConvImpl` + `DWConvBlockImpl`, `legacy` flag on `DetectImpl`, `infer_model_info` C2PSA marker detection.
- ✅ Two parity-resolving fixes: BN eps=1e-3 (was libtorch default 1e-5), v26 SPPF differences (cv1 act=False + residual shortcut).
- ✅ Two task-head fixes: pose keypoint decode (`xy*2*stride + (anchor_pix − 0.5*stride)`), OBB rotated decode (angle-aware `dist2rbox`).
- ✅ Multi-label NMS + `[0, orig]` box clamp.

### 1.16 yolo12 (Phase 6C)
- ✅ Detect end-to-end — train / val / predict / ONNX + TRT export across n/s/m/l/x.
- ✅ Forward parity-clean (5/5/5/6/5 dets matching Python on bus.jpg).
- ✅ ONNX max\|Δ\| ≤ 1.78e-7 vs Python.
- ✅ `AAttn` (per-head qkv interleave fix) + `ABlock` + `A2C2f`.
- ✅ Gamma residual gate at l/x (caught `300 detections at conf=0.25` saturation).
- ✅ `pe.conv.bias=True` opt-in flag in `ConvImpl` for v12.

### 1.17 yolo13 (Phase 6D)
- ✅ Detect end-to-end across n/s/l/x (no `m` upstream — iMoonLab fork).
- ✅ Train / val / predict / ONNX + TRT export.
- ✅ Forward cls-channel max\|Δ\| ≤ 7.6e-10 vs iMoonLab Python.
- ✅ ONNX max\|Δ\| ≤ 1.76e-7.
- ✅ HyperACE module set: `DSConv` / `DSBottleneck` / `DSC3k` / `DSC3k2` / `DownsampleConv` / `FullPADTunnel` / `FuseModule` / `AdaHyperedgeGen` / `AdaHGConv` / `AdaHGComputation` / `C3AH` / `HyperACE`.
- ✅ V13-specific `V13AAttn` / `V13ABlock` / `V13A2C2f` (separate qk/v convs, k=5 pe).

### 1.18 yolo26 (Phase 6B)
- ✅ Full 5 scales × 5 tasks — train / val / predict / ONNX + TRT export.
- ✅ STAL assigner + ProgLoss training.
- ✅ DFL-free Detect head, end-to-end NMS-free inference.

### 1.19 Cross-cutting infrastructure
- ✅ `engine::validate<M>` templated runner (any holder with `forward_eval(x) → [B, 4+nc, A]`). Explicit instantiations for v3/v4/v5/v6/v7/v8/v9/v10/v11/v12/v13/v26 (task #17).
- ✅ `engine::TrainerT<M>` templated trainer with default `LossTraits<M>` → `V8DetectionLoss`, specialised `LossTraits<Yolo26Detect>` → `Yolo26Loss`.
- ✅ Versioning + changelog policy (pre-1.0, `0.MINOR.PATCH`) — `CHANGELOG.md`, `CMakeLists.txt` `VERSION`, CLAUDE.md policy section.
- ✅ Periodic gap-audit (recurring TODO) — task #33 + CLAUDE.md `## Periodic gap-audit` section.

---

## 2. Pending / in-flight tasks (session task numbers)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #19 | v3..v10 ONNX/TRT export — CLI dispatch + per-version scoping | ✅ closed | n/a | split into #34..#39 |
| #34 | v3 ONNX/TRT export | ✅ closed | n/a | landed in 0.14.0 — 7 dets via TRT FP32 matching C++ |
| #35 | v4 ONNX/TRT export | ✅ closed | n/a | landed in 0.15.0 — 6 dets via TRT FP32 matching C++ |
| #36 | v6 ONNX/TRT export — all 12 variants | ✅ closed | n/a | n/s in 0.17.0; m/l in 0.19.0 (BepC3+DFL+SimSPPF); P6 + MBLA in 0.21.0 (4-level head + MBLABlock with BottleRep3). bus.jpg TRT FP32: P5 n/s/m/l=4/5/5/6, MBLA s/m/l/x=6/6/6/5, P6 n6/s6/m6/l6=5/6/5/8 dets — all matching libtorch. |
| #37 | v7 ONNX/TRT export | ✅ closed | n/a | landed in 0.16.0 — base 5 dets, tiny 0.90 conf matching C++ |
| #38 | v9 ONNX/TRT export — all 5 scales | ✅ closed | n/a | t/s/m/c in 0.14.1; e (CBLinear+CBFuse two-pass backbone) added in 0.20.0; v9{c,e} TRT FP32 returns 5 dets matching libtorch. |
| #39 | v10 ONNX export (one2one head) | ✅ closed | n/a | landed in 0.7.1 |
| #40 | v10 TRT FP32 detection drop on s/m/b/l/x scales | ✅ closed | n/a | landed in 0.18.0 — root cause was three bugs: CLI scale default of "n" mis-built the engine; CLI11 export/predict/val skipped filename-based scale resolution; TRT default-on TF32 saturated v10 cls outputs. Fix: clear `kTF32` for v10 builds + auto-resolve scale. All 6 scales now match ORT (5 dets on bus.jpg). |
| #41 | v10 train (one-shot from #32 deferral) | ✅ closed | n/a | landed in 0.13.0 (one2one head) |
| #45 | v10 dual-head consistent-assignment training | ✅ closed | n/a | landed in 0.22.0 — added `Yolo10Impl(scale, nc, dual_head)` ctor with parallel `o2m_detect` (legacy=true v8-style cv3), `losses::V10DualLoss` (V8DetectionLoss with topk=10 + topk=1), `losses::Yolo10LossAdapter` runtime branch, `convert_yolov10_dual_pt` to load both heads' pretrained weights, CLI `dual_head=true` flag, and `tests/test_v10_dual_train.cpp` smoke (loss 16.85→15.14 in 4 steps on coco8, 432 tensors loaded). |
| #21 | v10 s / m / b / l / x predict + val | ✅ closed | n/a | landed in 0.7.0 |
| #23 | v6 MBLA variants — s_mbla / m_mbla / l_mbla / x_mbla | ✅ closed | n/a | landed in 0.8.0 |
| #24 | v6 high-res P6 — n6/s6/m6/l6 all parity-clean | ✅ closed | n/a | n6/s6/m6 in 0.9.0; l6 closed in 0.18.1 |
| #42 | v6l6 parity gap — saturated cls at conf=0.25 | ✅ closed | n/a | landed in 0.18.1 — root cause was BN eps=1e-5 vs upstream 1e-3 + V6ActScope leaking SiLU into structural neck convs (reduce_layer/Bifusion/downsample) which upstream hardcodes as ReLU. Backbone+neck now bit-exact vs Python; bus.jpg gives 8 dets at conf=0.25. |
| #31 | v6 train (VFL + SIoU + TAL) | ✅ closed | n/a | landed in 0.11.x; mAP@0.5:0.95=0.74 finetune on coco8 |
| #43 | v6 train scaffolding (V6DetectionLoss + TrainerV6) | ✅ closed | n/a | architecture in 0.11.0 |
| #44 | v6 train TAL parity — fg_mask=0 on first iter | ✅ closed | n/a | one-line fix in 0.11.1 |
| #29 | v3 train (yolov3u) | ✅ closed | n/a | landed in 0.10.0 |
| #30 | v4 / v7 train (anchor-based v3-style loss) | ✅ closed | n/a | landed in 0.12.0 — V7DetectionLoss handles both |
| #31 | v6 train (VFL + SIoU + TAL) | medium | 2 sessions | new loss class — anchor-free with KD `reg_preds_dist` as DFL target |
| #32 | v10 train (single-head one2one) | ✅ closed | n/a | landed in 0.13.0 — V8DetectionLoss on one2one head |
| #33 | Codebase gap audit | recurring | 30–60 min per pass | none — runs at task-batch / phase boundaries, or on demand |

---

## 3. Pending — by version

### yolo3
- ✅ All wired (predict + val + train + ONNX/TRT) for the yolov3u variant.
- ❌ Original Darknet anchor-based head — would need a `.weights` binary parser + LeakyReLU activation switch. Out of scope.

### yolo4
- ✅ All wired (predict + val + train + ONNX/TRT).

### yolo6
- ✅ All 12 variants (n/s/m/l + 4×_mbla + n6/s6/m6/l6) wired end-to-end (predict + val + train + ONNX/TRT).
- ❌ Lite / face variants (out of scope).

### yolo7
- ✅ All 7 variants (base/tiny/x/w6/e6/d6/e6e) predict + val + ONNX/TRT export. Train wired for base.
- ❌ Tiny / x / w6+ train — would need per-variant anchor configs (reuse V7DetectionLoss).
- ❌ Auxiliary head (`IAuxDetect`) for original training-time PGI behaviour — stripped at deploy upstream.

### yolo9
- ✅ All 5 scales (t/s/m/c/e) wired end-to-end (predict + val + train + ONNX/TRT).
- ❌ PGI auxiliary branch in train (intentionally not wired — training-only upstream; would yield a few mAP points).

### yolo10
- ✅ All 6 scales (n/s/m/b/l/x) wired end-to-end (predict + val + train + ONNX/TRT).
- ✅ Paper §3.1 dual-head consistent assignment training (0.22.0).

### yolo12 / yolo13
- ⏳ Task heads (segment / pose / obb / classify) — neither Ultralytics nor iMoonLab publishes task weights upstream. **Planned future session:** train our own on COCO using the existing templated `Trainer`. v12 = 5 scales × 4 tasks = 20 runs; v13 = 4 scales × 4 tasks = 16 runs. v12 task scaffolding exists in `src/models/yolo12_tasks.cpp`; v13 task module declarations not yet written.

### RT-DETR
- 🟡 Architecture probed (Phase 4 — transformers). Header / source files exist; `mode=predict` returns a clear "not implemented yet" error pointing at the design plan.

---

## 4. Code-level TODOs / FIXMEs

These live as inline comments in the codebase. Each should either be tracked as a task above or be explicitly accepted as won't-fix.

| location | note | mapped to |
|----------|------|-----------|
| `include/yolocpp/datasets/yolo_dataset.hpp:20` | `Mosaic / mixup are TODO.` | §5 (training augmentations) |
| `src/tasks/segment_train.cpp:388` | `pull feats explicitly (TODO: expose forward_train_seg).` | §5 (segment trainer cleanup) |

Note: the `legacy stub holder` in `src/models/yolo26.cpp` and the `Object stub` in `pt_loader.cpp` are not TODOs — they're intentional sentinels.

---

## 5. Cross-cutting / infrastructure

These don't map to a single YOLO version.

- ❌ **Mosaic / mixup augmentation** — straightforward but ~600 lines of C++ we haven't needed yet. Touches `datasets/yolo_dataset.hpp`.
- ❌ **AMP (mixed-precision training)** — Trainer is FP32-only. Adding `torch::autocast` + `GradScaler` is a future change.
- ❌ **Multi-threaded data prefetch** — dataset is synchronous. With OpenCV decode + CUDA inference, the IO bottleneck on a 5090 is real but fixable later.
- ❌ **TRT INT8 calibration** + dynamic-shape multi-batch profiles — easy on top of `TrtBuildConfig` once a calibration set exists.
- ❌ **Two-GPU DDP validation** — wiring is in place + world_size=1 verified, but no two-GPU box has run training yet.
- ❌ **`forward_train_seg` factor-out** — segment trainer currently reaches into the segment head to pull feats; a clean accessor would let it ride the same templated trainer pattern as detect.
- ❌ **Benchmark CLI for non-detect tasks** — currently detect-only; classify/seg/pose/obb benchmarks are a small extension.

---

## 6. Maintenance protocol

**Update this file as part of every work batch.** When a task lands:
1. Move the line from §2 / §3 / §5 to §1 (Completed work) under the appropriate sub-section.
2. Add a CHANGELOG.md entry under a new `## [X.Y.Z]` heading.
3. Bump `project(yolocpp VERSION X.Y.Z)` in CMakeLists.txt.
4. If the change affects user-visible behaviour, refresh README.md / CLAUDE.md.

Recurring task **#33** (gap audit) periodically re-walks this file, the capability matrix in CLAUDE.md, the README front-matter, the per-file `TODO`/`FIXME` grep, and the SKIP-gated tests — see CLAUDE.md `## Periodic gap-audit (recurring TODO)` for the full checklist and trigger points.

---

## 7. Pre-1.0 disclaimer

This is a **pre-1.0** project. Current version: see CMakeLists.txt `VERSION`. The public API, on-disk weight format, CLI surface, and dataset conventions may all still change. The 1.0 line is gated on the maintainer's call, not a feature checklist.
