# TODO — yolocpp full task ledger

**Single source of truth for everything done, in flight, and pending across the entire codebase** — not just the active session. Compiled by walking session task #1..#33, README.md "deferred" sections, CLAUDE.md "Remaining gaps", per-file `TODO`/`FIXME`/`deferred` comments, SKIP-gated tests, and the git history (Phases 0..6 pre-dating session task numbering).

This file is maintained as part of recurring task **#33** (gap-audit) — see CLAUDE.md `## Periodic gap-audit (recurring TODO)` for the audit checklist and trigger points.

The current release version is **always read from `CMakeLists.txt` `project(... VERSION ...)`** (which flows into `build/generated/yolocpp/config.hpp` as `YOLOCPP_VERSION_STRING` and out via `yolocpp info`). Do not duplicate it into prose snapshots in this file — the only places a literal version belongs are `CMakeLists.txt`, `CHANGELOG.md` headings, and historical "landed in X.Y.Z" lines.

> **Latest snapshot**: every numbered task #12..#45 closed except recurring #33. Full matrix sweep (`scripts/full_matrix_sweep.sh`) reports `PASS=152 FAIL=0 SKIP=0`. ctest 31/31 green. New roadmap tasks **#46..#63** below were filed 2026-05-01 from the user's next-batch requirement list. See `SESSION_DIGEST.md` for the per-version landing map across the previous session.

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
- ✅ Clean-room pickle parser for upstream `.pt` (handles every opcode `torch.save` produces; treats unknown GLOBAL classes as opaque object stubs while still extracting tensor data).
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
- ✅ All 75 (version × task × scale) combinations for v8/v11/v26 load the upstream-shipped weights and produce non-empty output on bus.jpg.
- ✅ Task ONNX export — graph emitters for segment proto / pose kpt decode / OBB dist2rbox / classify head. ONNX max\|Δ\| ≤ 0.004 across tasks.

### 1.5 Production-training extras (Phases 3.2 – 3.8)
- ✅ Auto-resolve `model=` and `data=` from cwd / cache / upstream asset URL.
- ✅ Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied).
- ✅ LR-warmup formula fixed for tiny datasets.
- ✅ Trainer saves `.pt` in `load_state_dict`-compatible format.
- ✅ Scale auto-detected from filename.
- ✅ Save-dir auto-increments (`runs/train` → `train2` → `train3`).
- ✅ `best.pt` saved at peak val mAP@0.5:0.95.
- ✅ Auto-attach val split when `<root>/images/val` exists.
- ✅ `results.csv` per-epoch (upstream-shape header).
- ✅ `patience=N` early stopping when val mAP plateaus.
- ✅ `runs/<run>/args.yaml` reproducibility dump (timestamped, 107-key upstream-shape).
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
- ✅ upstream-style `data=path/to/data.yaml` (yaml-only — no directory form).
- ✅ `data.yaml` parsed via vendored rapidyaml (`path:` / `train:` / `val:` / `names:` / `download:`).
- ✅ Auto-download from `data.yaml`'s `download:` URL when dataset missing.
- ✅ Model auto-inference from `.pt` state_dict shapes — version, scale, nc.
- ✅ `scale=` / `version=` / `nc=` no longer required (works on renamed `best.pt` / `last.pt`).

### 1.8 yolo3 (Phase 5B + task #13 + #29)
- ✅ Predict end-to-end (upstream `yolov3u`) — Darknet-53 + v8 anchor-free DFL head. 103M params; 7 dets on bus.jpg.
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
- ✅ Full-COCO val mAP@0.5:0.95 within 0.05% of the upstream `m.val(rect=False)` on n/s.
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

## 2A. Roadmap — next-batch requirements (filed 2026-05-01, tasks #46..#63)

Filed in priority order. Tasks are grouped so dependent items land together. Sub-tasks (`#NA`/`#NB`/...) belong to the same parent and are resolved as a unit. Optional / nice-to-have items live at the end of the list and don't block anything.

**Ordering convention going forward:** when a new gap turns up while resolving task `#N`, file it as `#NA`, `#NB`, … (related + dependent → same parent). When it's unrelated, append it to the end of the queue with the next free number. Never delete tasks from this file unless the user asks; superseded items get crossed out (`✅ closed` / `❌ won't fix`) but stay on the page.

### Group I — foundation (must land before downstream groups)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #46  | Modular architecture for adding new YOLO versions in one pass | high | 2 sessions | export pipeline migrated; predict/val/train follow as #46D/#46E/#46F |
| #46A | ✅ closed — `include/yolocpp/registry/version_adapter.hpp` + `src/registry/version_registry.cpp` provide a `VersionAdapter` (std::function-typed hooks) and a `Registry` singleton seeded by `register_all_versions()`. All 12 versions register themselves declaring `version_id`, `display_name`, `default_export_basename`, `supported_tasks`, `default_imgsz(scale, task)`, `export_onnx(...)`, and a `trt_disable_tf32` quirk flag. | — | landed | — |
| #46B | ✅ closed (interim) — type erasure via std::function in `VersionAdapter` is the abstract base every concrete `Yolo<N>Impl` plugs into without inheritance. A real C++ concept-based base (one ABC every Impl satisfies via `forward_train` + `forward_eval` shapes) is deferred to #46H once predict/train are also on the registry; not blocking. | — | landed | — |
| #46C | ✅ closed — "how to add a new YOLO version" walkthrough lives at the top of `include/yolocpp/registry/version_adapter.hpp` (4-step recipe: drop the model TU, write a `register_yolo<N>` helper, add it to `register_all_versions()`, list the TU in CMakeLists). Mirrored in CLAUDE.md. | — | landed | — |
| #46D | ✅ closed — `cmd_predict_task` migrated. `VersionAdapter::predict_to_file` hook routes to the per-version `inference::predict_v<N>_to_file` helpers; v8 leaves the hook empty and falls back to the unified `inference::Predictor`. Replaced ~120 lines of if-else with a single registry call. Test extended (`tests/test_registry.cpp`) asserts every non-v8 adapter wires `predict_to_file`; smoke verified with v11 (5 dets) + v8 fallback (6 dets) on bus.jpg. | — | landed | — |
| #46E | ✅ closed — `cmd_val` migrated. `VersionAdapter::run_val` hook constructs the right holder, calls `engine::validate<M>`, and returns `(map_50, map_50_95)`. v8 leaves it empty and falls back to the unified `inference::Predictor` (Yolo8Detect-only, the path's only architecture). v3..v10 + v11/v12/v13/v26 now correctly route through their concrete holder types — previously the fallback misloaded everything except v8 + v11/v26-via-Predictor's accidental shape match. Test extended (`tests/test_registry.cpp`); end-to-end val run on coco8 with yolo11s gives mAP@0.5=0.949 / mAP@0.5:0.95=0.707. | — | landed | — |
| #46F | 🟡 partial — `cmd_train` (kv-style detect) migrated; `VersionAdapter::run_train_detect` constructs the holder, optionally `load_state_dict`s init weights, runs the matching `TrainerT<Holder>` (`LossTraits<M>` specialisation does the per-version loss binding for free). v8 falls back to `engine::Trainer` as before. Verified on coco8 1-epoch smoke: v11s mAP@0.5:0.95=0.58, v8l (fallback) 0.69. **Pending:** `engine::run_benchmark` per-version dispatch (`engine/benchmark.cpp:158-242`) — same shape, follow-up commit. | — | within #46 | — |
| #46F2 | ✅ closed — `engine/benchmark.cpp` now consults the registry. `build_onnx_for` reduced from ~95 lines to a single `adapter->export_onnx(...)` call (reuses the export hook from #46A). `bench_pt` reduced from ~95 lines to `adapter->benchmark_pt(...)`. New `include/yolocpp/engine/benchmark_internal.hpp` factors out `GenericPredictor<Holder>` + `bench_one()` so registry TUs can use them. v8 still falls back to the legacy `inference::Predictor`. Smoke: yolo11s on bus.jpg → PT 4.98 ms, TRT FP32 2.91 ms (1.71×), TRT FP16 2.40 ms (2.07×); 5 dets across all three backends. `test_benchmark` (ctest) passes. | — | landed | — |
| #46F3 | ✅ closed — `dispatch_kv` val + train chains collapsed. The val branch (~170 lines reimplementing v3..v11/v26 dispatch) → single `cmd_val()` call. The train branch (~286 lines reimplementing v3..v11/v26 inline trainers) → single `cmd_train()` call, with v10 dual-head case kept inline (its `Yolo10(scale, nc, dual_head)` ctor isn't on the standard adapter signature). **Net deletion: 432 → 19 lines.** Smoke verified: kv-style `task=detect mode=val model=yolo11s.pt data=coco8/data.yaml` gives mAP@0.5=0.949 / 0.5:0.95=0.707 (identical to pre-refactor numbers); 1-epoch train via kv style finishes with mAP@0.5:0.95=0.70. ctest 7/7 green. | — | landed | — |
| #46G | Add a per-version registry test (`tests/test_registry.cpp`) — asserts every expected `version_id` is registered and the minimum hooks exist. | ✅ closed | landed | — |
| #46H | ❌ won't fix — evaluated 2026-05-02 after #46A..F2 landed. **Decision:** keep the std::function-erased hooks. Reasoning: (a) `VersionAdapter` reads as a simple options-bag pattern; readers grok it instantly. (b) `static_cast<bool>(hook)` is a clean "did this version opt in?" check that an ABC would either replace with `std::optional<...>` returns (same semantics, more code) or with required virtual methods (forces every version to implement every hook even when the legacy fallback is correct — breaks v8's empty-hook pattern). (c) Compile-time signature checks already happen at every callsite that invokes a hook; the only added safety would be at registration time, which `tests/test_registry.cpp` already covers at runtime (one test, every version, every hook). (d) No measurable runtime cost: 12 entries × 5 hooks = 60 std::function instances, all hot-path-cached after first dispatch. The concept-based version would add ~12 classes, increase template instantiation, and not catch a single bug we don't already catch. Re-open only if a concrete failure mode motivates it (e.g., a future "every adapter MUST implement X" invariant the registry can't express today). | won't fix | n/a | — |
| #47  | Centralise version stamp — single source of truth | high | 0.5 session | low risk; mostly text edits |
| #47A | ✅ closed — top-level `./VERSION` file is now the single source of truth; CMake `file(READ)`s it into `project(... VERSION ...)`, exports through `config.hpp` (`YOLOCPP_VERSION_STRING`), surfaces via `yolocpp --version` / `-v` / `-V` and `yolocpp info`. To bump the version, edit `./VERSION` only. | — | landed | — |
| #47B | ✅ closed — SESSION_DIGEST.md re-headered as "frozen snapshot of prior session" (its `0.X.Y` mentions are now explicitly historical). README and CLAUDE already cleaned in earlier commit. One stale parenthetical in TODO.md re-worded to "landed in 0.22.0". | — | landed | — |
| #47C | ✅ closed — `scripts/check_version_literals.sh` added. Walks tracked files, flags any `0.X.Y` outside the allow-list (`./VERSION`, `CHANGELOG.md`, `SESSION_DIGEST.md`, historical "landed in X.Y.Z" / "added X.Y.Z" qualifiers, third-party / vendor strings). Currently passes; wire into pre-commit / CI when those land. | — | landed | — |
| #48  | ✅ closed — `third_party/DEPS.md` is the single pinned manifest (libtorch 2.11.0+cu130, TensorRT 10.14.1.48+cuda13, CUDA 13.0.88, OpenCV 4.6.0, NCCL 2.23.4, rapidyaml 0.11.1, CLI11 2.4.x). `scripts/audit_deps.sh` enforces it: any unknown `find_package` or undocumented `third_party/` directory fails the audit. Currently passes (4 packages, 8 dirs). The "no Boost / no protobuf / no GTest / no fmt / no json / no ORT" rationale is documented inline; "how to add a new dep" checklist included. | — | landed | — |
| #48A | ✅ closed — covered by #48 (`third_party/DEPS.md`). | — | landed | — |
| #48B | ✅ closed — audited; no redundant deps. Every `find_package` / `target_link_libraries` is genuinely consumed; the manifest documents why. | — | landed | — |
| #48C | (optional) Trim `third_party/{tensorrt_root,opencv_root}` `.deb` extractions to only the components we link against — would shave ~2.6 GB. Touches `install_third_party.sh`'s extraction logic; not done now (disk pressure isn't blocking). | low | within #48 | — |
| #49  | ✅ closed — every code-level mention of the upstream vendor neutralised across CLI / models / inference / engine / serialization / datasets / tasks / tests / scripts. Allow-list (kept by design): (a) the `kAssetBase` URL constant in `cli/resolve.cpp` (real network endpoint that hosts the legacy weights — documented as the only allow-listed mention); (b) the Meituan v6 release URL in `cli/resolve.cpp` (same reason); (c) the pickle wire-format token `"ultralytics.nn.tasks"` / `"DetectionModel"` in `pt_save.cpp` (downstream readers expect this exact GLOBAL — renaming would break interop); (d) historical entries in `CHANGELOG.md` (immutable history); (e) the `#49` task description itself (meta-reference). `looks_like_ultralytics_weight` → `looks_like_upstream_weight`. Help text, doc-comments, README/CLAUDE prose all read "upstream" / "the upstream form" / "upstream-shape" now. | — | landed | — |
| #49A | ✅ closed — within #49. CLI rename + identifier slice. | — | landed | — |
| #49B | ✅ closed — within #49. URL resolver allow-list documented in `cli/resolve.cpp` comment. | — | landed | — |
| #49C | ✅ closed — within #49. README / TODO / CLAUDE / SESSION_DIGEST prose neutralised. CHANGELOG left as historical record. | — | landed | — |
| ~~#50~~ | Pick + apply a license — **moved to Group VII (optional / deferred)**. Maintainer's call; recommendation on file is Apache 2.0 (patent grant covers ML/transformer IP, all upstream model deps already Apache/MIT, monetisation works via dual-licensing + hosted SaaS + premium weights). Filed at #50 originally; no longer blocks #60 in the ordering — but #60 *publish* must wait until a license is chosen. | optional | — | — |

### Group II — CLI / API surface (depends on Group I)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #51  | CLI overhaul — clean, intuitive, both long + short forms | high | 2 sessions | depends on #49 (legacy strings) |
| #51A | ✅ closed — every CLI11 subcommand now registers long+short forms: `-m/--model` (alias `--weights`), `-d/--data`, `-s/--source`, `-o/--out`, `-D/--device`, `-i/--imgsz`, `-e/--epochs`, `-b/--batch`, `-n/--nc`, `-c/--conf`, `-f/--format`. Help text now describes every flag. Legacy `--weights` kept as a CLI11 alias on every subcommand. Verified: `yolocpp predict -m data/yolo11s.pt -s data/bus.jpg -D auto` writes 5-dets jpg; `yolocpp benchmark -m ... -s ... -D auto` produces full PT+TRT timing. | — | landed | — |
| #51B  | ✅ closed — both kv-style and legacy subcommand-style removed under #51K. | — | landed | — |
| #51B2 | ✅ closed — folded into #51K (full removal, not phased). | — | landed | — |
| #51J  | ✅ closed — flat flag-style CLI added as canonical. | — | landed | — |
| #51K  | ✅ closed — kv-style (`task= mode= model= ...`) and legacy subcommand-style (`yolocpp train -m ...`) parsers REMOVED entirely per maintainer's request. Flag-style (`yolocpp --mode <action> ...`) is now the only CLI parser. Deleted: `dispatch_kv`, `looks_like_kv_style`, `looks_like_mode_style` (no longer needed — flag-style is unconditional), the entire CLI11 subcommand block, dead helpers (`task_implemented`, `kSupportedTasks`, `kSupportedModes`), `src/cli/args.cpp`, `include/yolocpp/cli/args.hpp`. Help text + file header refreshed. Both removed styles now reject with `--mode is required`. All 7 modes (predict / val / train / export / benchmark / info / download) verified through the canonical parser. | — | landed | — |
| #51C | 🟡 partial — `--source` now classifies to one of {Image, Dir, Glob, Video, Url, Webcam, Unknown}. Image / Dir / Glob fan out to a per-image loop; Video / Url / Webcam surface a clear "deferred to #51C2" error with an ffmpeg-split workaround. Unknown specs error before any I/O. Smoke verified across all six paths. | — | within #51 | — |
| #51C2 | ✅ closed — `inference::FramePredictor` polymorphic base + `make_frame_predictor` adapter hook on every non-v8 version (uniform `GenericFramePredictor<Holder>` wrapper around `engine::detail::GenericPredictor`). v8 routes through an inline `V8FramePredictor` wrapping the unified `inference::Predictor`. CLI's `cmd_predict_task` opens `cv::VideoCapture(spec)` for Video/URL/Webcam (one API handles all three), runs `pred->predict(frame)` per decoded frame, draws via the new shared `inference::draw_detections`, writes annotated mp4 to `runs/predict/<stem>.mp4` via `cv::VideoWriter`. Webcams cap at 600 frames so a forgotten Ctrl-C doesn't fill the disk. Smoke verified: 30-frame synth video → v11 (registry) 150 total dets / 700 KB out, v8 (fallback) 162 dets / 695 KB out. URL streams that can't be opened reject cleanly. Test extended (`tests/test_registry.cpp`) asserts every non-v8 adapter wires `make_frame_predictor`. | — | landed | — |
| #51D | ✅ closed — `TrainConfig::seed` (uint64_t, default 0). When non-zero seeds `torch::manual_seed`, `torch::cuda::manual_seed_all`, the trainer's batch-sampler RNG, and (transitively via sampler state) per-example augmentation RNGs. CLI surface: `yolocpp train --seed=42` (CLI11) and `yolocpp ... mode=train seed=42` (kv-style). Seed reflected in `<save_dir>/args.yaml`. Caveat documented in `TrainConfig::seed` docstring: GPU runs aren't bit-deterministic by default (cuDNN + atomic adds yield ~1% per-batch loss noise); the strict-deterministic flag is queued as #51D2. | — | landed | — |
| #51D2 | Add an opt-in `--strict-deterministic` flag that calls `torch::globalContext().setDeterministicAlgorithms(true, /*warn_only=*/false)` so GPU runs become bit-reproducible at the cost of ~10–30% throughput. Document the trade-off. | low | within #51 | — |
| #51E | ✅ closed — `yolocpp download <name\|url>` subcommand. Known short-names: `coco8`, `coco8-seg`, `coco8-pose`, `coco128`, `coco128-seg`, `dota8`, `VOC`, `xView` (registry table in `cli/resolve.cpp`). Direct URLs (any `*://*`) supported as fallback. Already-present datasets short-circuit with a "already present" log. Idempotent: re-running on an existing target is a no-op. Adding a new dataset is two lines in `kKnownDatasets`. Tested: `download coco8` → no-op (already there); `download not-a-real-dataset` → clear error listing the known names. | — | landed | — |
| #51F | 🟡 partial — `yolocpp export --precision/-p {fp32\|fp16\|int8\|int4\|nvfp4}`. fp32 / fp16 wired through the existing `TrtBuildConfig.fp16` toggle. int8 / int4 / nvfp4 surface a clear "not yet wired (#51F2)" error. Legacy `--fp16/--no-fp16` kept as alias. Bad values rejected with a list of acceptable forms. | — | within #51 | — |
| #51F2 | INT8 calibration + INT4 / NVFP4 routing. INT8 needs a per-architecture calibration set (probably reuse coco8 val); INT4 / NVFP4 require Blackwell-era TRT 10.4+ APIs. Document workspace_bytes increase for INT8. | medium | within #51 | — |
| #51G | ✅ closed — `yolocpp train --export-after-train=onnx[,trt]`. After train completes, auto-exports `<save_dir>/best.pt` (falls back to `last.pt`) into `<save_dir>/best.{onnx,trt}` so the artefact lives next to the run. Pre-resolves scale + version_hint from the init weights filename so the post-train export doesn't mis-shape an unsuffixed `best.pt`. Smoke verified: 1-epoch yolo11s coco8 train → `best.onnx` (37.8 MB) lands alongside `best.pt`. | — | landed | — |
| #51H | ✅ closed — centralised `normalise_device(d)` helper validates `cpu`, `cuda`, `cuda:N`, `cuda:0,1,…`, `mps`, `auto` (empty / `auto` → "" so callers pick CUDA-when-available). Bad indices and unrecognised forms throw with a clear message before any subcommand body runs. Wired into both the kv-style and CLI11 paths. Smoke verified: `--device=cuda:99 → "[error] --device=cuda:99: index 99 out of range (have 1)"`. | — | landed | — |
| #51I | 🟡 partial — top-level `yolocpp --help` rewritten with concrete copy-pasteable examples for every subcommand (predict/val/train/export/benchmark/download/info), kv-style example block kept, and "Tasks / Modes / Formats / Devices / Source" cheat-sheet at the bottom. Per-subcommand help (`yolocpp predict --help` etc.) reads cleanly via the per-flag descriptions added in #51A. **Pending:** consistent exit-code audit (currently mixes `2` for user error, `1` for runtime; uniform standard would be 0 ok / 2 user error / 1 runtime — tracked as #51I2). | — | within #51 | — |
| #51I2 | Exit-code audit — settle on 0 success, 2 user error (bad flags, unknown values), 1 runtime error. Sweep every `return 1` / `return 2` site in `cli/main.cpp` and align. | low | within #51 | — |
| #52  | ✅ closed — public chainable C++ API. `yolocpp::YOLO("yolo11s.pt").to("auto").predict({.source = "bus.jpg"})` etc. Mirrors the upstream Python ergonomics (`predict`, `val`, `train`, `export_`, `benchmark` methods on a `YOLO` class; designated-initialiser Args structs per method). Every method routes through `yolocpp::cli::cmd_*` so behaviour is identical to the CLI. Required a refactor: `cmd_*` bodies moved out of `main.cpp`'s anonymous namespace into `src/cli/commands.cpp` (in `yolocpp_core`) so both the CLI driver and the API can link against them; `cmd_dispatch_flag_style` stayed in `main.cpp` as it's the only CLI11 consumer. `tests/test_api.cpp` exercises image-mode predict, dir fan-out, export, and chaining; passes. | — | landed | — |
| #52A | ✅ closed (interim) — image-mode `predict()` returns `std::vector<Detection>` (currently empty; the on-disk artefact is the source of truth). Future enhancement: thread per-image dets back through the `predict_to_file` registry hook so the API can return populated vectors without re-running inference. Filed as #52A2. | — | landed | — |
| #52A2 | ✅ closed — `VersionAdapter::predict_to_file` now returns `vector<Detection>` (was `size_t`). All 11 lambdas updated. `cmd_predict` + `predict_one_image` + `cmd_predict_task` thread an optional `out_dets` pointer through; `YOLO::predict()` populates it and returns the dets. For multi-input dir/glob runs the API gets the LAST image's dets — per-image map is filed as #52A3. test_api extended to verify det count + bbox sanity. | — | landed | — |
| #52A3 | Per-input det map for dir/glob predict via the API. `YOLO::predict_many({.source = "frames/"})` would return `vector<vector<Detection>>` keyed by sorted filename instead of just the last. Defer until a use case asks. | low | within #52 | — |
| #52B | Add `examples/` with usage snippets (single image, video, custom training, export pipeline). | low | within #52 | — |
| #52C | (deferred) optional pybind11 wrapper. Off the runtime path; only for users who want Python bindings on top of the C++ API. | optional | within #52 | — |

### Group III — verification (run continuously once Group I + II land)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #53  | ✅ closed — strict cross-backend parity ctest (#53A) + per-version TRT round-trip in the sweep (#53B). | — | landed | — |
| #53A | ✅ closed — `tests/test_cross_backend_parity.cpp`. For each cell: export ONNX through the registry, build TRT FP32 + FP16 engines, run libtorch / TRT-FP32 / TRT-FP16 on bus.jpg, assert det count within ±1 of libtorch and IoU ≥ 0.50 same-class match for ≥ N-1 of N libtorch dets. Cells: v8n / v11s / v12s / v13s — 4/4 pass with `matched N/N`. v26 deliberately excluded (NMS-free deploy form's output shape doesn't unpack through the standard TrtPredictor post-process; covered by `test_v26_e2e` separately). Engines cached in `build/parity_cache/` for repeat speed. | — | landed | — |
| #53B | ✅ closed — `scripts/full_matrix_sweep.sh` phase 7 added: per-version `.pt → .trt → predict` round-trip across all 12 supported versions. **Sweep total now 164/164 PASS (was 152)** — adds the 12 trt-roundtrip cells. Catches export-emitter / TRT-parser / runtime regressions at sweep-time. | — | landed | — |
| #54  | ✅ closed — dataset infra v2: flat-format + COCO JSON + Pascal VOC loaders, mAP S/M/L breakdown, Mosaic-4 + Mixup. | — | landed | — |
| #54A | ✅ closed — `FlatDataset`. Single CSV/TSV with header `split,image_path,class_id,x_center,y_center,width,height`. Auto-detects delimiter; `--seed` for deterministic shuffle. | — | landed | — |
| #54B | ✅ closed — `VocDataset` + `CocoDataset`. VOC reads `JPEGImages/Annotations/ImageSets`; XML via regex (no libxml2). COCO reads `instances_<split>.json` via a hand-rolled JSON tokenizer (no libjson per `DEPS.md`). Sparse COCO category IDs compressed to dense [0, N). Both yield the standard `YoloExample`. | — | landed | — |
| #54C | ✅ closed — `metrics::mAPResult` extended with `map_50_95_{small,medium,large}` + `n_gt_*` counts. Three additional passes per call (COCO area buckets ≤32², ≤96², >96²). Surfaced through `cmd_val` to stdout + `runs/val/<stem>_results.txt`. Verified on coco8 (4/7/6 GT counts; mAP 0.06/0.37/0.88 on yolo11s). | — | landed | — |
| #54D | ✅ closed — Mosaic-4 + Mixup in `YoloDataset::sample_batch`. `AugConfig::mosaic_p`/`mixup_p` (default 0). Mosaic stitches 4 sampled images at a random centre, crops to imgsz×imgsz, drops boxes that shrink to ≤1 px. Mixup blends with α~Beta(8,8) (8-sample uniform-ratio approximation). | — | landed | — |
| #54E | (optional follow-up) Factor `letterbox + hsv_jitter + label-pixel-remap` shared between YoloDataset / FlatDataset / VocDataset / CocoDataset into one helper. ~80 lines duplicated 4× today; pending until a fifth loader makes it painful. | low | within #54 | — |

### Group IV — feature add-ons (independent of one another; can land in any order after Group I)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #55  | Trackers + SAHI integration | medium | 3 sessions | needs a clean abstract `Tracker` base; design first |
| #55A | Centralised abstract `Tracker` base (`update(det) → tracks`) | — | within #55 | — |
| #55B | SORT | — | within #55 | — |
| #55C | DeepSORT (re-id embedder) | — | within #55 | — |
| #55D | OC-SORT | — | within #55 | — |
| #55E | ByteTrack | — | within #55 | — |
| #55F | BoT-SORT | — | within #55 | — |
| #55G | NvSORT | — | within #55 | — |
| #55H | SAHI (slicing-aided hyper inference) wrapper around `Predictor` for small-object recall | — | within #55 | — |
| #56  | Add legacy / additional YOLO families (depends on #46 modularisation) | medium | many sessions | each variant is its own self-contained sub-task |
| #56A | yolo1 — implement architecture from paper; convert pretrained weights to our `.pt`-equivalent (no Darknet) | — | within #56 | — |
| #56B | yolo2 / yolo9000 — same approach (no Darknet) | — | within #56 | — |
| #56C | YOLOX | — | within #56 | — |
| #56D | YOLO-NAS | — | within #56 | — |
| #56E | YOLO-WORLD (open-vocab) | — | within #56 | — |
| #56F | YOLOE | — | within #56 | — |
| #56G | YOLOR | — | within #56 | — |
| #56H | PP-YOLO / PP-YOLOE | — | within #56 | — |
| #56I | Scaled-YOLOv4 | — | within #56 | — |
| #56J | DAMO-YOLO | — | within #56 | — |
| #56K | Centralised model zoo umbrella for non-YOLO open-source CV models that share the same license profile (commercial-friendly) | — | within #56 | — |

### Group V — performance + hardware

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #57  | Parallelisation pass over hot paths | medium | 1 session | profile-driven; not a blanket change |
| #57A | Multi-threaded data prefetch (existing TODO in §5) | — | within #57 | — |
| #57B | Audit + parallelise per-image preprocess / NMS / export emitters where safe | — | within #57 | — |
| #57C | CUDA streams overlap for multi-batch predict | — | within #57 | — |
| #58  | Multi-device + cross-platform deployment (depends on #51H for CLI) | medium | 3 sessions | per-platform sub-tasks |
| #58A | CPU / multi-CUDA / MPS device dispatch in core | — | within #58 | — |
| #58B | iPhone / iOS deployment (CoreML export) | — | within #58 | — |
| #58C | Android deployment (NNAPI / TFLite via ONNX) | — | within #58 | — |
| #58D | Small ARM SBC edge devices | — | within #58 | — |
| #59  | Jetson + DGX Spark TRT export profiles | medium | 1 session | requires the device for actual validation |
| #59A | Jetson Nano TRT plan | — | within #59 | — |
| #59B | Jetson Orin TRT plan | — | within #59 | — |
| #59C | Jetson THOR TRT plan | — | within #59 | — |
| #59D | DGX Spark TRT plan | — | within #59 | — |

### Group VI — distribution + documentation

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #60  | Retrain every (version × scale × task) on COCO; publish weights to GitHub Releases | medium | many sessions | depends on #50 (license decided) and #54 (dataset infra) |
| #60A | Train script harness driving the templated trainer across the matrix | — | within #60 | — |
| #60B | Compute budget plan (GPUs × hours per cell) | — | within #60 | — |
| #60C | Release artifact upload pipeline | — | within #60 | — |
| #60D | Mirror the resulting weights table in README + CLI auto-resolver | — | within #60 | — |
| #61  | Comparison table + graphs ("which model when") | medium | 1 session | depends on #60 numbers |
| #61A | mAP / params / FLOPs / latency table per (version × scale × task) | — | within #61 | — |
| #61B | Auto-generated SVG graphs in `docs/` | — | within #61 | — |
| #61C | Decision-tree style picker ("pick a model for my use case") | — | within #61 | — |

### Group VII — optional / nice-to-have (do not block anything)

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #62 | Optional: Ninja generator support for faster builds | low (optional) | 0.25 session | none; just `cmake -G Ninja` validation + docs |
| #63 | Optional: cross-platform GUI (Dear ImGui / Qt) for train/val/predict/export | low (optional) | many sessions | not on the critical path |
| #50 | Optional: license decision (Apache 2.0 recommended). Moved here from Group I per maintainer — no quick decision wanted. Re-promote when the maintainer is ready to commit to a license; gates #60 *publication* but not #60 training itself. | optional | 0.25 session | — |
| #64 | `tests/test_v6_e2e.cpp` lines 64 / 97 fail to compile — `predict_v6_to_file` signature took an `NMSConfig` previously but now expects a different type (compiler reports "cannot convert NMSConfig to bool"). Pre-existing breakage discovered during #46 — surfaced when the full build was kicked off, not caused by the registry refactor. Fix: update test to match current `predict_v6_to_file` signature, or restore the helper. | medium | 0.25 session | — |

---

## 2B. RF-DETR family (#65 — landed scaffold 2026-05-02 in 0.31.0)

First non-YOLO architecture. Transformer-based DETR-family detector
(encoder/decoder + object queries + Hungarian-matching set loss +
DINOv2 / LW-DETR backbone). Scaffold is registered in the registry
under `version_id="rfdetr"`; every CLI / API path dispatches to a
slice-tagged throw until the corresponding sub-task lands. Variants:
`rfdetr-{n,s,b,m,l}.pt` for detect, `rfdetr-{n,s,b,m,l}-seg.pt` for
segment.

| # | scope | priority | session-cost estimate | blockers |
|---|-------|----------|------------------------|----------|
| #65   | Top-level RF-DETR family (umbrella for #65A..#65L). Scaffold landed 0.31.0; all entry points throw with slice tags. | high | many sessions | — |
| ~~#65A~~  | ~~Backbones: DINOv2 (ViT-L) + LW-DETR-{tiny,small,base,medium}.~~ Landed 0.32.0 — `models/rfdetr_backbone.{hpp,cpp}` shared `PatchEmbed`/`Attention`/`ViTBlock`/`ViTBackbone`; submodule names match upstream so the #65D converter is a prefix rename. `tests/test_rfdetr_backbone.cpp` pins the n/s/b/m forward shapes (DINOv2-large skipped on memory). | done | — | — |
| ~~#65B~~  | ~~Transformer encoder (multi-scale deformable attention, sine pos-emb).~~ Landed 0.33.0 — `rfdetr_encoder.{hpp,cpp}` MSDeformAttn as a portable grid_sample composition + per-level 1×1 input proj + 2D sine pos-embed. `tests/test_rfdetr_encoder.cpp` pins n/s output shapes (b/m/l skipped on memory). | done | — | — |
| ~~#65C~~  | ~~Decoder + object-query head.~~ Landed 0.34.0 — `rfdetr_decoder.{hpp,cpp}` MLP/DecoderLayer/DetrHead; vanilla self-attn + MSDeformAttn cross-attn + iterative bbox refinement; `[B, 4+nc, Q]` YOLO contract. `forward_eval` end-to-end; `forward_train` returns per-layer cls+bbox for #65F. Tests: `test_rfdetr_forward.cpp`. | done | — | — |
| ~~#65D~~  | ~~`.pt` converter — investigation phase.~~ Landed 0.38.0 — downloaded all 12 official 1.6.5 weights, dumped per-key shape inventories, wrote `docs/rfdetr_arch.md` ground-truth spec, registered all 12 variants in `RFDetrScale` with real hyperparameters, taught the CLI resolver to recognise `rf-detr-*.pth` / `rf-detr-seg-*.pt`. **The actual converter is blocked on the architecture rewrite** since the scaffold's module layout doesn't match upstream key names. Tracked as #65A2..F2 below. | done (investigation) | — | — |
| ~~#65A2~~ | ~~Replace `rfdetr_backbone.{hpp,cpp}` with HF-DINOv2 windowed-attn.~~ Landed 0.40.0 — full Dinov2{PatchEmbeddings,Embeddings,SelfAttention,SelfOutput,Attention,LayerScale,MLP,Layer,Encoder,Model,Wrapper,WrapperOuter} with separate Q/K/V, layer_scale1/2, 2D-interpolated pos embed. RFDetrImpl registers as `backbone.0.encoder.encoder.*` matching upstream key names exactly. **Loader binds 223/487 keys** from `rf-detr-base.pth` (full backbone). | done | — | — |
| ~~#65B2~~ | ~~Add `rfdetr_projector.{hpp,cpp}` (CSP-style).~~ Landed 0.41.0 — `ConvBN/ProjBottleneck/ProjStage0/Projector/BackboneSlot`; `n_proj_stages=2` for large; per-variant `pretrain_grid`+`backbone_embed`. **Loader binds 248-273 / 487-600 keys across all 12 variants**, full backbone+projector. CLI scale resolver now extracts `rf-detr-large` etc. correctly. | done | — | — |
| ~~#65C2~~ | ~~Rewrite decoder.~~ Landed 0.42.0 — `rfdetr_transformer.{hpp,cpp}` FusedMHA / MSDeformAttn1L / RFDetrMLP / RFDetrDecoderLayer / RFDetrDecoder + ref_point_head + shared cls/bbox heads + refpoint_embed/query_feat embeddings. **Detect-n/s/m/b bind 100% from upstream**; large needs 4-level cross-attn (#65C2-large follow-up). | done | — | — |
| ~~#65D2~~ | ~~Two-stage encoder-output.~~ Landed 0.42.0 — `RFDetrTransformer` siblings `enc_output / enc_output_norm / enc_out_class_embed / enc_out_bbox_embed` (13 entries each, group_detr). | done | — | — |
| #65C2-l | Large variant cross-attn dims: 4-level × 3-point deformable (currently 1×2); plus extra `stages_sampling` projector block. | medium | 0.25 session | — |
| #65E2 | `rfdetr_weights.{hpp,cpp}` — direct loader from upstream `.pt` / `.pth`. **Partial:** loader landed in 0.39.0 (handles flat-dict pickle, runs full `--mode predict` path through random-init scaffold; reports matched/unmatched coverage). Promotion to `strict=true` blocked on #65A2..D2 module renames. | partial (0.39.0) | 0.25 session remaining | #65A2..D2 |
| ~~#65F2~~ | ~~Real-arch forward end-to-end.~~ **Done 0.60.0** — full backbone + projector + two-stage + iterative-decoder + heads + NMS-free decode produces real detections from upstream `rf-detr-*.pth` weights. detect-base: 5/5 dets on bus.jpg at conf=0.7. fp64 enc-output for deterministic topk. CLI hook now actually wires through `rfdetr_predict_image`. n/s/m/b verified working; large pending #65C2-l. | done | — | — |
| ~~#65E~~  | ~~Predict path / NMS-free decode.~~ Landed 0.36.0 — `inference/rfdetr_predictor.{hpp,cpp}::rfdetr_decode` (threshold + top-K) + `rfdetr_predict_image` (letterbox → forward → decode → unscale). Tests: `test_rfdetr_decode.cpp` against synthetic forward tensor. Registry hook still throws at converter (#65D). | done | — | — |
| ~~#65F~~  | ~~Hungarian-matching set loss.~~ Landed 0.35.0 — `losses/hungarian.{hpp,cpp}` Jonker-Volgenant rectangular assignment + `losses/rfdetr_loss.{hpp,cpp}::rfdetr_set_loss` (focal cls + L1 + GIoU, auxiliary loss across decoder layers, last-layer matching). Tests: `test_rfdetr_loss.cpp`. Independent of #65D — works against random-init forward. | done | — | — |
| ~~#65G~~  | ~~Trainer integration.~~ Landed 0.37.0 — `LossTraits<RFDetr>` specialization in `engine/trainer.cpp` (de-interleaves flat forward_train, converts YOLO `[N,6]` tgt → per-image RFDetrTarget lists). `TrainerRFDetr` typedef. `forward_eval` now returns xyxy-pixels per YOLO contract. Registry hook calls TrainerRFDetr::run instead of throwing. | done | — | — |
| #65H  | Validator + per-area mAP (S/M/L per #54). Wire `run_val` adapter hook. Drop-in via existing `engine::Validator` once forward_eval lands. | high | 0.25 session | #65E + #54 mAP S/M/L |
| #65I  | ONNX emitter: `src/serialization/rfdetr_onnx.cpp::export_rfdetr_onnx`. Standard transformer ops (MatMul, Softmax, Add, LayerNorm, Gather, Gemm) — mostly reusable; deformable-attn op needs a custom decomposition into per-level grid_sample (no `MultiscaleDeformableAttention` op in stock onnxruntime). Wire `export_onnx` adapter hook. | high | 1 session | #65D |
| #65J  | TRT pipeline: route through shared `build_trt_engine`; verify TF32 on Blackwell sm_120 doesn't regress query attention numerics. Wire `benchmark_pt` adapter hook. | high | 0.5 session | #65I |
| #65K  | Segment variant: per-query mask coefficients + shared proto bank (à la YOLACT, mirrored from YOLO's segment head); wire `RFDetrSegment` forward + load_state_dict + a `-seg` filename suffix. | high | 1 session | #65E + segment-task infra |
| #65L  | Per-variant parity smokes: `tests/test_rfdetr_e2e.cpp` (SKIP-gated when weights missing) + ONNX-vs-Python forward comparison through onnxruntime CPU on a fixed `arange/(N-1)` input; cls max\|Δ\| ≤ 1e-3 detect, ≤ 5e-3 segment. | medium | 0.25 session | #65I,K |

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
- ✅ Paper §3.1 dual-head consistent assignment training (landed in 0.22.0).

### yolo12 / yolo13
- ⏳ Task heads (segment / pose / obb / classify) — neither the upstream maintainer nor iMoonLab publishes task weights upstream. **Planned future session:** train our own on COCO using the existing templated `Trainer`. v12 = 5 scales × 4 tasks = 20 runs; v13 = 4 scales × 4 tasks = 16 runs. v12 task scaffolding exists in `src/models/yolo12_tasks.cpp`; v13 task module declarations not yet written.

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
