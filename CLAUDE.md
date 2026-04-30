# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

A pure-C++ replacement for Ultralytics. **No Python at runtime.** LibTorch for
training/eval, TensorRT for deployment, OpenCV for image I/O.

### Supported YOLO versions (closed set)

The project covers exactly twelve YOLO versions, with **no `v`** in any
filename, identifier, namespace, class name, comment, or doc string:

```
yolo3   yolo4   yolo5   yolo6   yolo7   yolo8
yolo9   yolo10  yolo11  yolo12  yolo13  yolo26
```

Anything outside this set (v1, v2, v14..v25, v27+) is **not supported and
not planned**. When patching code, follow the convention everywhere — even
when referencing legacy upstream Ultralytics URLs that still publish as
`yolov<N>...pt`. The single legitimate place that strings differ from the
canonical form is `src/cli/resolve.cpp::upstream_basename`, which maps a
canonical local name back to the upstream URL when downloading v3..v10.

### Implementation status

`yolo8` and `yolo11` are fully end-to-end (train / val / predict / ONNX +
TRT export across all 5 scales × 5 tasks). `yolo5` is end-to-end via the
anchorless `*u.pt` variants. `yolo3` has the architecture in place
(forward-shape verified; weight loader deferred). All other versions
exist as **stubs** under `src/models/yolo<N>.cpp` +
`include/yolocpp/models/yolo<N>.hpp` — the header carries the design
intent; the source throws a clear `not implemented yet` from `forward()`.
Implementation order — see README.md — prefers families that reuse the
v8 Detect / DFL / TAL stack (yolo9/10) before the ones needing new heads
(yolo26) or new state dict adapters (yolo4/6/7).

### Known issue: v11m/l/x cv3 saturation

The yolo11m/l/x scales load bit-exactly (param counts and tensor shapes
match Ultralytics published values exactly) and produce realistic
detections for high-salience objects, but their cls outputs are
systematically over-saturated. Empirical evidence from a forensic
investigation:

- v11n/s: zero-input → 0 saturated (sigmoid > 0.5) cls outputs.
- v11m/l/x: zero-input → 822–1198 saturated cls outputs.

This is independent of input signal — even constant-zero input produces
hundreds of saturated outputs for these scales. The amplification
originates in the cv3 (cls-branch) of the Detect head. v11m's cv3 BN
scales (`bn.weight / sqrt(running_var + ε)`) are 4–6× larger than v11n's
in absolute mean. Large BN scales aren't pathological per se (they
reflect small training-time activation variance), but they amplify any
deviation from the running mean — and the bias chain through cv3
produces a non-zero output that gets cumulatively amplified.

The bug reproduces in the TRT engine path too (built from our
hand-emitted v11 ONNX), so it's in the forward *graph*, not in
libtorch's separate Conv+BN ops.

**Investigated and ruled out:**
- Weight loading (verified bit-exact via shape/key set comparison)
- Backbone forward (layers 0–10 produce healthy magnitudes for all scales)
- GPU vs CPU (same behavior on both)
- FP32 vs FP16 inference (same)
- Sequential vs ModuleList for C3k's inner `m`
- Force-c3k=True logic for m/l/x (necessary — checkpoint has C3k
  structure at all C3k2 layers; otherwise load fails with shape mismatch)
- C3k forward order (`a = m(cv1(x))`, `b = cv2(x)`, `cat({a, b})`)

**Likely root cause:** subtle forward-path numerical divergence specific
to the m/l/x configuration (8 c3k=True C3k2 layers + max_channels=512
cap) that compounds across the chain. The structural ops match
Ultralytics' published code; the divergence is at the numerical level.

**Unblocking:** a Python parity harness — load yolo11m.pt in Ultralytics'
Python, dump per-layer intermediate tensors for a fixed input
(`bus.jpg`), then run the same input through our C++ and compare
element-wise. The first divergence layer is the bug. This tooling
matches the "Phase 1.5" parity validation already noted in the
"Parity validation" section above and would also benefit other yolo
versions when they're added.

**Workaround:** for v11m/l/x, the model still works for high-confidence
detections of salient objects — predict produces both `person` and
`bus` correctly on `bus.jpg`. The mAP penalty on coco8 is real but the
model isn't broken — just over-confident. Increasing conf threshold
doesn't help (saturated outputs persist at conf=0.9+); a confidence
calibration pass against Ultralytics output would be needed.

### v11-specific notes

The v11 build introduced new shared infrastructure that all subsequent
families can reuse:
- `DWConvImpl` (depthwise conv) and `DWConvBlockImpl` (DWConv → Conv 1×1)
  in `yolo8.hpp`. The latter exists because libtorch's `nn::Sequential`
  cannot hold another `Sequential` (templated forward breaks AnyModule);
  DWConvBlock provides a non-templated forward and child names `"0"`/`"1"`
  matching Ultralytics' state_dict layout.
- A `legacy` flag on `DetectImpl` (default `true` for v3/v5/v8/v9) — when
  `false`, cv3 builds the v11 nested `(DWConv→Conv)→(DWConv→Conv)→Conv2d`
  form instead of the legacy `Conv→Conv→Conv2d`.
- For YOLO11 m/l/x scales, Ultralytics' `parse_model` overrides every
  C3k2's `c3k=True` regardless of YAML; we replicate this with
  `if (scale.width_multiple >= 1.0) c3k = true;` in the v11 yaml-walker.
- `infer_model_info` detects v11 via the C2PSA marker
  (`model.{9,10}.m.0.attn.qkv.conv.weight`) since the stem-channel
  table alone is ambiguous for some scales (v8l and v11m both have stem
  ch=64). When stem ch=64 + PSA marker present, we further disambiguate
  v11m vs v11l by the depth signature (`model.6.m.1` exists for l, not m).

## Build, test, run

```bash
./scripts/install_third_party.sh                         # one-time, ~5 GB
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure               # all tests
ctest --test-dir build -R smoke_test --output-on-failure # one test by regex
./build/tests/smoke_test                                 # run a test directly
./build/yolocpp info                                     # CLI sanity check
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `89;90;120` and `TORCH_CUDA_ARCH_LIST`
to `8.9;9.0;12.0` — set those if targeting different hardware. nvcc must be
on PATH (or pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`).

## Toolchain — non-obvious choices

- **LibTorch is `cu130`**, not `cu128`. The system has CUDA 13.0 toolkit at
  `/usr/local/cuda` and PyTorch ships matching cu130 builds. Don't "downgrade"
  to cu128 unless you also reinstall the toolkit.
- **TensorRT install skips `libnvinfer-dev`** (the 2.9 GB deb). It only
  contains static libs we don't link against. The unversioned `.so` symlinks
  the linker needs are produced by `scripts/install_third_party.sh` after
  extracting the runtime debs.
- **OpenCV is vendored from Ubuntu 24.04 universe debs**, not built from
  source. The runtime `.so.4.6.0` files come from `libopencv-*406t64`; the
  headers + CMake configs come from `libopencv-*-dev`. Both are extracted
  under `third_party/opencv_root/` and exposed via `third_party/opencv/`
  symlinks.
- **Blackwell (sm_120) requires** TRT ≥ 10.13 with the cuda13.0 build. The
  smoke test only passes if `libnvinfer_builder_resource_sm120.so.10.14.1`
  is present in `third_party/tensorrt/lib/`.

## The DT_RPATH gotcha — do not remove

Every executable uses `target_link_options(... LINKER:--disable-new-dtags)`
so the linker emits **DT_RPATH** (transitive) instead of DT_RUNPATH
(non-transitive). Reason: `libnvinfer.so` calls `dlopen()` on its
sm-specific resource libraries (`libnvinfer_builder_resource_sm{75,80,86,89,90,100,120}.so.10.14.1`)
at engine-build time. With DT_RUNPATH the executable's rpath does **not**
propagate to those dlopen calls and you get:

```
IBuilder::buildSerializedNetwork: Error Code 6: API Usage Error
(Unable to load library: libnvinfer_builder_resource_sm120.so.10.14.1)
```

If you add new executables, give them the same `target_link_options` and
`BUILD_RPATH`/`INSTALL_RPATH` (use the `_yolocpp_rpath` variable defined in
the root CMakeLists).

## Architecture (planned, mostly unimplemented)

The directory layout under `src/` and `include/yolocpp/` mirrors the
Ultralytics decomposition: `backbones/`, `necks/`, `heads/`, `losses/`,
`datasets/`, plus `engine/` (training loop), `export/` (ONNX + TRT),
`core/` (device/version/utility), `cli/` (CLI11-based entry point).

The architectural commitments are:
- **One unified training engine** across all model families.
- **One unified export pipeline** (ONNX + TRT) across all models.
- Each task (detect / segment / pose / OBB / classify) = head + loss +
  dataset format + postproc; the rest is shared.

Phase 1 built yolo8n end-to-end first (architecture, weight loader from
Ultralytics `.pt` checkpoints, dataset loader, training loop, validation,
inference, ONNX/TRT export). Phase 6 added yolo11 end-to-end alongside
v8 (re-using v8's loss / trainer / validator templates, with new
modules — C3k2 / C2PSA / Attention / PSABlock / DWConvBlock). Subsequent
phases scale to the other YOLO versions in the closed set above, then to
transformer-based detectors (RT-DETR, ViT).

## CLI surface

`yolocpp {train,val,predict,export,info}` — only `info` is implemented.
Stubs return exit code 2. When implementing a subcommand, the entry point
is `src/cli/main.cpp`; add the implementation under the matching `src/`
subdirectory and link it into `yolocpp_core`.

## Editing third_party/

Files under `third_party/` are produced by `scripts/install_third_party.sh`
and are gitignored. **Don't commit anything in there.** If a deb extraction
needs to change (different TRT version, different OpenCV mirror), update the
script — not the extracted tree.

## Parity validation (Phase 1)

Numerical parity against Ultralytics is a hard requirement before any model
ships. The reference dumps will be produced by a one-off Python tool kept
**outside the build** (dev-only, not in the runtime path). When that lands,
parity tests will live in `tests/parity_*.cpp` and gate forward-pass / loss
implementations against snapshotted tensors.
