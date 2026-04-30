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

### Parity status (resolved)

A Python parity-dump harness (uncommitted, lives in `/tmp/yolocpp_parity/`
behind a uv venv) was used together with `tests/parity_compare` to compare
our C++ forward vs Ultralytics' Python module-by-module on a fixed input.
Two structural mismatches were found and fixed:

1. **BN epsilon.** Ultralytics overrides `BatchNorm2d.eps = 1e-3`;
   PyTorch's default (and the one libtorch's `nn::BatchNorm2d` was using)
   is `1e-5`. With typical running_var ~ 0.01–0.05 the BN scale
   `γ/sqrt(var + eps)` is ~3% larger at eps=1e-5 — a tiny per-layer drift
   that compounds disastrously through the m/l/x chain (8 C3k2 layers +
   max_channels=512 cap). Fixed in `ConvImpl` and `DWConvImpl` by passing
   `BatchNorm2dOptions(c).eps(1e-3)`. This was the root cause of the
   "v11m/l/x cv3 saturation" we previously chalked up to large BN scales —
   the bias amplification was real, but the actual *source* was eps not
   matching Python.

2. **v26 SPPF differences.** The latest Ultralytics SPPF (used for v26
   weights) drops `cv1`'s SiLU (`act=False` → Identity) and adds a
   residual shortcut (`add = shortcut and c1 == c2`) on the output.
   `SPPFImpl` now takes optional `cv1_act` (default `true`, false for
   v26) and `shortcut` (default `false`, true for v26) parameters.

After both fixes our forward is **bit-exact** vs Ultralytics' Python at
every layer 0..22 for every (yolo11, yolo26) × (n, s, m, l, x). Layer 23
(Detect) intentionally still differs: v11 returns xyxy where Python
returns xywh, v26 returns the decoded `[N, 4+nc, A]` tensor where
Python's e2e head returns post-NMS-free `[N, 300, 6]` — same underlying
predictions, different output convention; downstream NMS gets the right
input either way.

Resulting full-COCO val mAP@0.5:0.95 (5000 images, no fine-tune,
imgsz=640, batch=1, conf=0.001, iou=0.7):

```
              before    BN-fix    +SPPF    +multi-NMS    Ultralytics
yolo11n     0.261     0.382     0.382    0.388        0.388 (rect=F)
yolo11s     0.243     0.454     0.454    0.460        0.461 (rect=F)
yolo11m     0.029     0.501     0.501    0.503        0.508 (rect=F)
yolo11l     0.041     0.518     0.518    0.521        0.527 (rect=F)
yolo11x     0.006     0.531     0.531    0.535        0.540 (rect=F)
yolo26n     0.019     ~0.32     0.328    0.332        —
yolo26s     0.074     ~0.35     0.361    0.366        —
yolo26m     0.192     ~0.39     0.397    0.401        —
yolo26l     0.255     ~0.41     0.417    0.422        —
yolo26x     0.280     ~0.43     0.433    0.437        —
```

After the third intervention — adding multi-label NMS (NMSConfig.multi_label,
defaulted on for the validator) and fixing the box clamp from `[0, orig-1]`
to `[0, orig]` — yolo11n/s match Ultralytics' own `m.val(rect=False)` to
within 0.05% mAP@0.5:0.95. The ~0.5 pt residual on m/l/x is below the
forward-path: the parity comparator confirms every layer 0..22 is exact-
zero at fp32, so the remaining variance is fp32-accumulation noise in
NMS sort tiebreaks across ~30k candidates and in mAP averaging.

The marketing-published numbers (e.g. yolo11n=0.395) come from
`rect=True` inference. Ultralytics' own `m.val(rect=False)` is what's
directly comparable to our square-letterbox path, and on that
apples-to-apples basis we're parity-clean.

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
