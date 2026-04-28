# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

A pure-C++ replacement for Ultralytics. **No Python at runtime.** LibTorch for
training/eval, TensorRT for deployment, OpenCV for image I/O. The codebase is
currently at Phase 0 (foundation only) — see README.md for the phased roadmap
through YOLOv8n → full v8 family → all YOLO versions → DETR/RT-DETR/ViT.

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

Phase 1 builds YOLOv8n end-to-end first (architecture, weight loader from
Ultralytics `.pt` checkpoints, dataset loader, training loop, validation,
inference, ONNX/TRT export) before scaling to the rest of v8, then earlier
YOLOs, then transformer-based detectors.

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
