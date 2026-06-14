# Session digest — 2026-06-14 (current)

End-state of the most recent session. For the live version see `./VERSION` /
`yolocpp --version`. ctest 40/40 green; work landed on branch
`claude/todo-sweep` (8 commits off `main`; not pushed). This session was an
audit-driven sweep through TODO §2A: closed several documented gaps, corrected
ledger drift, and advanced #70.

**Where the next session starts: the remaining large/blocked items.**
- **#5** benchmark for non-detect tasks (classify/seg/pose/obb) — needs
  task-aware PT/TRT predictors + per-task mAP (top-1 / mask-mAP / OKS /
  rotated-IoU); larger than the "small extension" the ledger implied.
- **#55** trackers (SORT/OC-SORT/ByteTrack/BoT-SORT/NvSORT + SAHI) — needs a
  clean abstract `Tracker` base first; multi-session.
- **#70** is now *partial*: the cv::dnn parse blocker is fixed (DFL `ReduceSum`
  → axes-attribute), but cv::dnn 4.6's forward still can't run the decode
  subgraph (internal shape assertion) — true ONNX mAP needs the onnxruntime dep
  (maintainer decision).
- Hard-blocked: v12/v13 task heads (#60, COCO training), #58/#59 deploy
  (devices), two-GPU DDP (hardware), #50 license, #61 publish.

| ver | scope | what landed |
|-----|-------|-------------|
| 0.101.3 | fix | **#47C2** version-literal linter passes (allow-list extended for historical/vendor/dep/comment forms) + wired as the `test_version_literals` ctest. |
| 0.101.4 | build+docs | **#57F** Ninja `STATUS` hint (ccache/mold auto-detect already wired); reconciled audit closeouts **#51D2** (`--strict-deterministic`, verified), **#52B** (`examples/` ships 7 snippets). |
| 0.101.5 | feature | **#51F2** INT8 export wired through the CLI + API (`-p int8 --int8-calib <dir>` → existing `ImgDirCalibrator`); int4/nvfp4 still reject. Verified yolo11n INT8 (5 dets, FP16 parity). |
| 0.101.6 | feature | **#52A3** `YOLO::predict_many()` → per-input dets keyed by path; `examples/predict_directory` demonstrates it. |
| 0.101.7 | fix | **#57G** `close_mosaic` actually wired (was advertised in `args.yaml` but never disabled mosaic) — trainer flips a shared dataset mosaic gate off for the last N epochs (mosaic+mixup); `--close-mosaic N` flag + short-run guard. |
| 0.101.8 | fix | **#70 (partial)** DFL `ReduceSum` → axes-attribute (cv::dnn-parseable, TRT bit-identical); benchmark ONNX predictor probes + degrades to an honest `-` (cv::dnn forward can't run the decode graph — needs onnxruntime). |
| 0.99.68–0.99.71 | data/fix | v5 resolver u-form 404 fix + v1 bf16 train-crash fix + models/ test skip-gating; 1-epoch metrics re-runs (batch-8 + baseline-matched batch-16) + delta CSVs under `docs/data/`; sweep hardened (`VARIANTS_FILE`, CUDA-OOM batch-halving, sci-notation/CRLF fixes). |
| 0.100.0 | feature | **`yolocpp_web`** browser console — server-side Clay→HTML UI + cpp-httplib backend (`src/web/`); jobs via the public `YOLO` API; no LibTorch in the browser. |
| 0.100.1 | feature | **`yolocpp::log`** — leveled, colour, friendly errors; `YOLOCPP_LOG` / `--debug`. resolve.cpp traced. |
| 0.100.2 | build | **deps centralized** in `cmake/dependencies.cmake` — CMake pins + pulls headers (`file(DOWNLOAD … EXPECTED_HASH)`) + LibTorch (FetchContent); vcpkg evaluated + rejected (no cu130 LibTorch / TensorRT port). |
| 0.101.0 | feature | **multi-model + format benchmark** — `-m a.pt,b.pt` → per-model format table + leaderboard; `--data` mAP; batch defaults to 1. |
| 0.101.1 | feature | **per-format mAP** — PT (validator) + TRT fp32/fp16/int8 (`engine::eval_predictor` over `TrtPredictor`, all versions, INT8 drop visible); ONNX via `cv::dnn` (graceful fallback — TODO #70). |

---

# Session digest — 2026-05-01 (previous)

Frozen snapshot of the prior session's end-state (the version stamp in
this header is **historical** — refers to that session's release). For
the current release version see `./VERSION` / `yolocpp --version`.

Session end-state: full matrix sweep `PASS=152 FAIL=0 SKIP=0`, ctest
31/31 green, every numbered task #12..#45 closed (except recurring #33
— gap audit). The "ver" column in the table below is the
landed-at-version of each line item (also historical / immutable).

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
