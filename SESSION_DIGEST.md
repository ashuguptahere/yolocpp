# Session digest — 2026-05-01

End-state: **yolocpp 0.24.0**, full matrix sweep `PASS=152 FAIL=0 SKIP=0`,
ctest 31/31 green, every numbered task #12..#45 closed (except recurring
#33 — gap audit).

## Versions shipped this session

| ver | scope | what landed |
|-----|-------|-------------|
| 0.17.1 | docs | post-0.17.0 audit cleanup (stale capability rows in README/CLAUDE.md) |
| 0.18.0 | fix #40 | v10 TRT FP32 cls saturation across all 6 scales — three bugs fixed: (a) CLI11 `scale_s` default of "n" mis-built engines for non-n weights, (b) cmd_export/cmd_predict/cmd_val skipped filename-based scale resolution, (c) TRT default-on `kTF32` saturated v10 cls (RepVGGDW 7×7 dwconv-with-bias accumulates mantissa loss). Added `TrtBuildConfig.tf32` field + per-version override. |
| 0.18.1 | fix #42 | v6l6 saturated cls — root causes: (a) BN eps=1e-5 vs Meituan's 1e-3 (verified across every released `.pt`), (b) V6ActScope leaking SiLU into structural neck convs which upstream hardcodes as ReLU. Fix: BN eps→1e-3 globally for v6 + nested `V6ActScope force_relu(false)` block in NeckImpl/NeckP6Impl. After fix v6l6 backbone+neck bit-exact vs Python (max\|Δ\|≤2e-5). All 12 v6 variants PASS. |
| 0.19.0 | feature | v6 m/l ONNX + TRT export (BepC3 + DFL) — added `emit_v6_bottlerep`, `emit_v6_repblockbr`, `emit_v6_bepc3`, `emit_v6_simsppf` emitters + DFL projection branch in `emit_v6_effidehead` (reshape→Softmax→Mul(proj=arange)→ReduceSum). bus.jpg TRT FP32 returns 4/5/5/6 dets across n/s/m/l. |
| 0.20.0 | feature | v9e ONNX + TRT export (CBLinear/CBFuse two-pass backbone) — 43-layer skeleton, `Identity` pass-through, `CBLinear` (1×1 Conv with bias), `CBFuse` (Slice + Resize(nearest, asymmetric) + Add per non-anchor input). v9e TRT FP32 = 5 dets matching libtorch. |
| 0.21.0 | feature | v6 P6 + MBLA ONNX + TRT export — closes the v6 export coverage gap (all 12 published variants now export). P6 path: 6-stage backbone, 4-level neck (3 reduce/Bifusion top-down + 3 downsample bottom-up), 4-input EffiDeHead. MBLA path: new `emit_v6_bottlerep3` + `emit_v6_mblablock` (Slice into branch_num chunks → run BottleRep3 Sequential capturing every intermediate → Concat → cv2). |
| 0.22.0 | feature #45 | v10 dual-head consistent-assignment training (paper §3.1) — added `Yolo10Impl(scale, nc, dual_head)` ctor + parallel `o2m_detect` (legacy=true v8-style cv3), `losses::V10DualLoss` (V8DetectionLoss with topk=10 + topk=1), `Yolo10LossAdapter` runtime branch, `convert_yolov10_dual_pt` to load both heads' pretrained weights, CLI `dual_head=true` flag, `tests/test_v10_dual_train.cpp` smoke. Loss 16.85→15.14 in 4 steps on coco8. |
| 0.22.1 | docs | post-#45 audit cleanup — stale `#40`/`#42`/`#45` references in CLAUDE.md/README.md/TODO.md rewritten. |
| 0.23.0 | feature | v5 ONNX + TRT export (closes last sweep failure) — new `emit_c3` helper (cv3(cat([cv1(x)→m→..., cv2(x)], dim=1))) reusing `emit_bottleneck` (which already handles arbitrary kernel sizes — v5's k=(1,3) Bottlenecks just work). 6×6 stem handled by generic `emit_conv_module`. New `export_yolo5_onnx` walks the 25-entry v5 yaml. All 5 v5 scales TRT FP32 = 4–5 dets matching libtorch. |
| 0.24.0 | feature | benchmark version dispatch — `engine::run_benchmark` previously hardcoded `Yolo8Detect`+`kYolo8n`. Now `BenchConfig.{version,scale,nc}` auto-resolve from filename; `GenericPredictor<ModelHolder>` template + `bench_pt`/`build_onnx_for` dispatch helpers handle all 12 versions. v10 TF32 cleared in TRT path. Sweep benchmark cells: 1 → 12, all PASS. |

## Final matrix sweep result

```
PASS = 152   FAIL = 0   SKIP = 0
```

| mode      | PASS | TOTAL | covers |
|-----------|------|-------|--------|
| predict   | 121  | 121   | every (version, variant, task) cell across the cache |
| val       |   4  |   4   | coco8 with v5n/v8n/v11n/v26n |
| train     |   3  |   3   | coco8 1-epoch batch=1 with v8n/v11n/v26n |
| export    |  12  |  12   | one cell per version (v3..v26) |
| benchmark |  12  |  12   | one cell per version (was v8-only pre-0.24.0) |

## Key architectural patterns established this session

1. **CLI scale auto-resolve** — `cli::scale_from_filename` is canonical;
   never default `scale_s` to "n" in CLI11 layer (silent mis-load
   bug class). Already wired across cmd_export, cmd_predict, cmd_val,
   `engine::run_benchmark`.

2. **`V10DualLoss` + `Yolo10LossAdapter`** pattern for dual-output
   models: a runtime adapter that branches on `feats.size()` lets the
   same `TrainerT<M>` template handle both single and dual outputs
   without specialization.

3. **`GenericPredictor<ModelHolder>`** template (in
   `src/engine/benchmark.cpp`) — reusable wrapper for any model with
   `forward_eval(Tensor)→Tensor`. Same letterbox→forward→NMS→
   scale_boxes pipeline as `inference::Predictor` but parametric over
   model type.

4. **`V6ActScope` nested-scope discipline** — `V6ActScope force_relu(false)`
   block inside Neck constructors keeps structural convs at ReLU even
   under outer `V6ActScope(true)` SiLU mode. Required because
   Meituan's neck hardcodes `ConvBNReLU` regardless of training_mode.

5. **TRT TF32 clear for v10** — unconditional across all 6 scales in
   both `cmd_export` and `run_benchmark`.

## Out-of-scope (won't fix per CLAUDE.md, NOT regressions)

- Original Darknet anchor-based v3 head (.weights binary loader).
- v6 lite/face variants.
- v7 IAuxDetect retrain.
- v9 PGI auxiliary branch.
- v12/v13 cls/seg/pose/obb weights — upstream ships only detect;
  scaffolding exists for v12 task heads in `src/models/yolo12_tasks.cpp`,
  v13 task headers not yet written. Future session: train our own task
  weights on COCO using existing `Trainer` template (5 v12 scales × 4
  tasks = 20 runs; 4 v13 scales × 4 tasks = 16 runs).

## What is NOT yet committed (working tree state)

71 files modified/added (`git status -s` shows them all). Notable
untracked:
- `CHANGELOG.md`, `TODO.md` (session-long ledgers)
- `scripts/full_matrix_sweep.sh` (dev-only matrix runner)
- `tests/test_v10_dual_train.cpp` + 14 other test files
- `tests/dump_*.cpp` (dev-only dumpers, not registered as ctest)
- `include/yolocpp/losses/yolo10_loss.hpp` + 5 other loss/serialization headers
- `src/losses/yolo10_loss.cpp` + 5 other src files
- `SESSION_DIGEST.md` (this file)

The user has NOT asked to commit. CLAUDE.md says "Only create commits
when requested by the user." Leave the working tree as-is for the next
session to decide.

## Reproducibility cheatsheet

```bash
# Build (~30s incremental)
cmake --build build -j$(nproc)

# Full ctest (~70s)
ctest --test-dir build --output-on-failure

# Full matrix sweep (~3min on CPU, builds 12 TRT engines)
bash scripts/full_matrix_sweep.sh
# → /tmp/yolocpp_full_sweep/{results.tsv,sweep.log,SUMMARY.md}

# Smoke check
./build/yolocpp info
./build/yolocpp task=detect mode=predict model=data/yolo11n.pt source=data/bus.jpg
```
