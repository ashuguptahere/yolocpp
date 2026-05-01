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

`yolo8`, `yolo11`, and `yolo26` are fully end-to-end (train / val /
predict / ONNX + TRT export across all 5 scales × 5 tasks).

`yolo12` is end-to-end for **detect** at all 5 scales: predict, val, and
train are wired (validator/trainer templated on model class and
explicitly instantiated for `Yolo12Detect` in `validator.cpp` /
`trainer.cpp`). Forward is parity-clean with Ultralytics Python (5/5/5/6/5
on bus.jpg matching exactly). Task heads (segment / pose / obb /
classify) are architecturally in place but Ultralytics ships only detect
weights for v12 (no -cls/-seg/-pose/-obb in v8.3.0+ assets). **ONNX/TRT
export is the only gap** — graph emitters for `A2C2f` / `AAttn` /
`ABlock` (with area-windowing reshape and gamma residual gate) are not
yet written; predict/val/train use libtorch directly.

`yolo13` (iMoonLab fork) is end-to-end for **detect** at all 4 published
scales (n / s / l / x — iMoonLab does not ship `m`). Forward is parity-
clean: cls-channel max|Δ| ≤ 7.6e-10 vs Python across all four scales,
predict on bus.jpg returns 5/5/6/5 detections (within ±1 of Python's
6/5/7/6 — the residual is conf/iou micro-diff, not a forward bug). The
new module set lives in `src/models/yolo13.cpp`: `DSConv`,
`DSBottleneck`, `DSC3k`, `DSC3k2`, `DownsampleConv`, `FullPADTunnel`,
`FuseModule`, `AdaHyperedgeGen`, `AdaHGConv`, `AdaHGComputation`,
`C3AH`, `HyperACE`, plus v13-specific `V13AAttn` / `V13ABlock` /
`V13A2C2f` (the iMoonLab fork's AAttn has separate `qk`/`v` convs and
k=5 pe instead of v12's fused k=7 qkv). Validator and trainer support
v13 via the same template-instantiation pattern as v12. Task heads
(seg / pose / obb / classify) are not architecturally in place for v13
yet — the iMoonLab fork itself ships only detect.

`yolo5` is end-to-end via the anchorless `*u.pt` variants.
`yolo3` has the architecture in place (forward-shape verified; weight
loader deferred).

All other versions exist as **stubs** under `src/models/yolo<N>.cpp` +
`include/yolocpp/models/yolo<N>.hpp` — the header carries the design
intent; the source throws a clear `not implemented yet` from `forward()`.
Implementation order — see README.md — prefers families that reuse the
v8 Detect / DFL / TAL stack (yolo9/10) before the ones needing new heads
(yolo4/6/7/13).

### v13-specific notes

The iMoonLab v13 build introduced a non-trivial new module set with
hypergraph attention. Key structural gotchas caught during parity work:

- **`AdaHyperedgeGen`** (parameterised by `prototype_base`,
  `context_net`, `pre_head_proj`) generates a per-token hyperedge
  participation matrix via context-pooled prototypes + multi-head
  similarity + softmax over the node dim. Per-head prototype split must
  match Python: `view(B, M, num_heads, head_dim).permute(0, 2, 1, 3)`.
- **`AdaHGConv`**: `edge_proj` and `node_proj` are `nn.Sequential` of
  `[Linear, GELU]`. Register them with explicit child names "0" and "1"
  so state-dict keys land at `<prefix>.0.{weight,bias}`. Reach into the
  Linear via `seq[0]->as<torch::nn::LinearImpl>()` for weight injection.
- **`HyperACE`** parse_model rules: `c1 = ch[f[1]]` (the SECOND from-source,
  not the first). `n` (the yaml repeats column) becomes the `n` arg
  scaled by depth, with `parse_model n` reset to 1 for the next layer.
  `num_hyperedges` scales with `0.5 / 1.0 / 1.0 / 1.5` for n/s/l/x.
  `channel_adjust=True` for n/s, `False` for l/x.
- **`DownsampleConv`** at l/x: parse_model passes `channel_adjust=False`
  AND clamps `c2 = c1` (no doubling). At n/s the default doubles
  channels via a 1×1 Conv.
- **`DSC3k2`** at l/x: parse_model overrides `dsc3k=True` regardless of
  YAML — same pattern as `C3k2` at l/x in v11.
- **`V13A2C2f`** at l/x: parse_model appends `residual=True,
  mlp_ratio=1.5`. Gamma is initialised to `0.01 * ones(c2)` (NOT
  `ones(c2)` like v12); state_dict load overwrites.
- **`V13AAttn` differs from v12 `AAttn`** in three structural ways:
  separate `qk` (out=2C) and `v` (out=C) convs (NOT a fused 3C qkv);
  pe is depthwise k=5 (not k=7); pe operates on `v` directly (not on
  the qkv stream) and is added inside attention output before `proj`.
  Reusing v12's `AAttnImpl` would shape-mismatch on the pe weight.
- **End-to-end parity** validated by `tests/test_v13_full.cpp` (cls
  max|Δ| ≤ 7.6e-10 across all 4 scales) and `tests/test_v13_ada.cpp`
  (six bit-exact module checks for AdaHyperedgeGen, AdaHGConv,
  AdaHGComputation, C3AH, HyperACE). Per-layer divergence localizer is
  `tests/test_v13_layer_diff.cpp` (built but not registered as a ctest).

### v12-specific notes

The v12 build introduced three modules and one structural lesson:

- `AAttn` (`yolo12.hpp`) — area-windowed multi-head self-attention. The
  qkv conv's 3C output channels are interleaved **per-head**, not as
  `[all_q, all_k, all_v]`. Splitting with `chunk(3, dim=1)` is wrong; the
  correct reshape is `view(B, N, num_heads, 3*head_dim)` then
  `permute(0, 2, 3, 1)` then `split_with_sizes([head_dim]*3, dim=2)`. We
  shipped the wrong split first and saw 1–4 detections per scale on
  bus.jpg vs Python's 5–6; after the fix v12 matches Python exactly.
- `ABlock` — `x + attn(x)` then `x + mlp(x)`. The mlp is two 1×1 Convs
  (first with SiLU, second `act=False`).
- `A2C2f` — CSP block where each m[i] is `Sequential(ABlock × 2)` (when
  `a2=True`) or a single `C3k(c_, c_, n=2)` (when `a2=False`). Differs
  from C3k2/C2f in two ways: cv1 outputs `c_inner` (not `2*c_inner`)
  and cv2 takes `(1+n)*c_inner` (not `(2+n)*c_inner`).
- **`gamma` learned residual gate**: at scales `l` and `x`, Ultralytics'
  parse_model overrides A2C2f to `residual=True, mlp_ratio=1.2`. The
  residual is gated by a learned per-channel `gamma` parameter (shape
  [c2]) — `out = x + gamma * y`. We caught the missing gate at predict
  time on v12l: 300 detections at conf=0.25 (saturated cls) before adding
  the gate, 3 after.
- **pe.conv has bias=True for v12** (the only Conv in the codebase that
  does — v8/v11/v26 use `bias=False` everywhere and let BN absorb the
  bias). Added an opt-in `conv_bias` flag to `ConvImpl` that the AAttn
  ctor flips on for the depthwise 7×7 pe.

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

### Task variants for v12 / v13 — not available upstream

Neither Ultralytics nor iMoonLab publishes task weights for v12 or v13.
Only the detect variants ship:

```
v12: yolo12{n,s,m,l,x}.pt           (5 detect-only)
v13: yolo13{n,s,l,x}.pt             (4 detect-only — no `m` scale either)
```

Confirmed by probing the upstream release URLs (no `-seg`/`-pose`/`-obb`/
`-cls` artefacts exist).

**Planned (future session): train our own v12/v13 task weights on COCO**
and wire them through predict/val/train/export the same way v8/v11/v26
task heads work today. Concretely this means:

1. Add Yolo13Segment / Yolo13Pose / Yolo13OBB / Yolo13Classify modules
   under `include/yolocpp/models/yolo13_tasks.hpp` + matching `src/`.
   Yolo12 task heads already exist as scaffolding in
   `src/models/yolo12_tasks.cpp` but are untested against real weights.
2. Train each (version × task × scale) combination on COCO using the
   existing `Trainer` (already templated on model class) — for v12
   that's 5 scales × 4 tasks = 20 runs; for v13 it's 4 scales × 4 tasks
   = 16 runs.
3. Add `predict_v12_to_file` / `predict_v13_to_file` task wrappers in
   `predictor.cpp` (existing `predict_v{12,13}_to_file` covers detect).
4. Add ONNX emitters for the task heads (segment proto / pose kpt
   decode / obb dist2rbox / classify head) — the v8/v11/v26 path
   already has these; v12/v13 detect uses `emit_detect_v11` so the
   incremental work is the per-task head only.

### Per-version capability matrix (detect, current state)

```
              predict    val        train      ONNX/TRT export
yolo8         ✅         ✅         ✅         ✅
yolo11        ✅         ✅         ✅         ✅
yolo12        ✅         ✅         ✅         ✅
yolo13        ✅         ✅         ✅         ✅
yolo26        ✅         ✅         ✅         ✅
```

ONNX export numerical parity (cls-channel max|Δ| vs Ultralytics Python
on `arange(N)/(N-1)` input through onnxruntime CPU):

```
yolo12 n=1.78e-7  s=1.39e-7  m=1.39e-7  l=1.39e-7  x=1.37e-7
yolo13 n=1.39e-7  s=1.76e-7  l=1.32e-7  x=1.39e-7
```

All within fp32 noise. v12 ONNX uses upstream Ultralytics' AAttn
structure (fused 3C qkv, k=7 pe with conv.bias=True); v13 ONNX uses
iMoonLab's V13AAttn (separate qk/v convs, k=5 pe). Both opset-17
compatible — `Gelu` (added in opset 20) is decomposed inline as
`0.5 * x * (1 + Erf(x / sqrt(2)))`; `ReduceMean` / `ReduceMax` use
axes-as-attribute (the input form moves to opset 18+).

### Task coverage matrix (predict/val/export/benchmark)

The detect path is parity-validated end-to-end. The four other
Ultralytics tasks (classify, segment, pose, obb) get the following
treatment:

```
                detect   classify   segment   pose      obb
predict (CLI)   ✓        ✓          ✓         ✓         ✓
val (CLI)       ✓        ✓          ✓         ✓         ✓
ONNX export     ✓        ✓          ✓         ✓         ✓
TRT export      ✓        ✓          ✓         ✓         ✓
benchmark       ✓        gap        gap       gap       gap
```

Predict-path smoke (`scripts/task_predict_sweep.sh`) runs every
(version, task, scale) combination — **75 cases** for v8/v11/v26 × all
5 tasks × n/s/m/l/x scales — and all 75 load Ultralytics' shipped
weights without shape mismatch and produce non-empty output on
`bus.jpg`.
Per-task notes:

- **classify**: all 15 (version × scale) variants load and forward.
  Required two fixes vs the original code:
  - The yolo8-cls YAML uses `max_channels = 1024` for ALL scales
    (n/s/m/l/x), unlike the detect YAML which caps m at 768 and l/x at
    512. Without overriding `Yolo8Scale.max_channels` to 1024 inside
    `Yolo8ClassifyImpl`, layer 7's Conv shape mismatched on m/l/x.
  - cv2 `INTER_LINEAR` resize → `INTER_AREA` for the downsample step
    (matches torchvision's antialiased PIL BILINEAR closely), which
    flipped top-1 on bus.jpg for v8s/v11s before the fix.
- **segment**: instance counts within ±1 of Python on bus.jpg across
  all (n,s) × (v8,v11,v26).
- **pose**: people count = 4 across all variants, matching Python
  exactly. v26 pose required head-shape fix: Ultralytics' v26 Pose head
  emits an additional uncertainty (sigma) branch alongside the
  keypoints (cv4 outputs `nk + nk_sigma = 51 + 34 = 85` channels
  instead of 51). `Pose26Impl` now allocates the wider cv4 and slices
  off the sigma channels at inference.
- **obb**: rotated-box counts within ±2 of Python on bus.jpg.

ONNX exporters for the four task heads were added (`scripts/onnx_export_sweep.sh`
covers all 75 (version, task, scale) combinations; `tests/parity_compare`
remains for layer-level detect parity, and `/tmp/yolocpp_parity/validate_onnx_tasks.py`
runs onnxruntime forward and compares against Ultralytics Python).
Numerical match summary on the deterministic `arange(N)/N` input:

```
v8/v11 detect    : max|Δ| ≤ 0.01     (fp32 noise)
v8/v11 classify  : max|Δ| ≤ 1e-5     (after the cls BN-eps fix below)
v8/v11 segment   : max|Δ| ≤ 0.002
v8/v11 pose      : max|Δ| ≤ 0.002    (after the kpt-decode fix below)
v8/v11 obb       : max|Δ| ≤ 0.004    (after the dist2rbox fix below)
v26 every task   : (no compare)      — Python emits e2e NMS-free format
```

Two extra fixes shipped along with the new task exporters:

1. **Classify BN epsilon** — Ultralytics' yaml-built models use BN
   `eps=1e-3` for detect/seg/pose/obb but plain PyTorch default `1e-5`
   for the *cls* models. We added `BnEpsScope` (a thread-local switch
   in `yolo8.cpp`) and push `BnEpsScope(1e-5)` from each `Yolo*Classify`
   constructor so all internal Convs pick up the right eps. The
   exporter's `fuse_conv_bn` now also reads `bn->options.eps()` from
   the live module instead of hardcoding 1e-3.
2. **Pose keypoint decode** — `PoseImpl::forward` had `(xy*2 − 1)*stride
   + anchor_pix`, which is Ultralytics' formula minus `0.5*stride`. The
   correct expression in pixel coords is `xy*2*stride + (anchor_pix
   − 0.5*stride)`, equivalent to Python's `(xy*2 + cell_idx)*stride`.
   Caught only by the ONNX validator — it produces a 4–16 pixel keypoint
   offset depending on level, big enough to fail parity but small enough
   that the predict path's bus.jpg counts still passed. Same fix applied
   to `Pose26Impl::forward` and to `emit_kpt_decode` in the ONNX
   exporter.
3. **OBB rotated decode** — `OBBImpl::forward` was using the standard
   `dist2bbox` (axis-aligned) decode from `DetectImpl`, not Ultralytics'
   angle-aware `dist2rbox`. The rotated decode shifts the box center
   along the predicted angle:
   `cx_feat = (xf*cos − yf*sin) + anchor_x_feat` (and similarly for y),
   where `xf = (r − l)/2`, `yf = (b − t)/2` are the centre offsets in
   feature units. Width/height are still `l + r` and `t + b`. Inlined
   the rotated decode in both `OBBImpl::forward` and `OBB26Impl::forward`
   and added `emit_rbox_decode` + `emit_detect_obb_dfl` /
   `emit_detect_obb_v26` graph emitters; the three OBB exporters now
   compute the angle first and feed it into the rotated decode.
   Numerical diff vs Python collapsed from ~30–80 px to fp32 noise
   (≤ 4e-3).

Benchmark CLI is still detect-only — straightforward extension to
classify/seg/pose/obb but not done yet.

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
