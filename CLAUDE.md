# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with this repository.

## Single source of truth: TODO.md

[`TODO.md`](TODO.md) is the canonical list of every task — completed,
pending, in-flight — for the entire codebase, not just the active session.
It aggregates session task numbers (#12..#45+), per-version capability
gaps, code-level `TODO` / `FIXME` comments, SKIP-gated tests, and
pre-numbered Phase work (Phases 0..6 from before the task numbering
scheme).

When the user asks "what's left?" / "show me all tasks" / "audit gaps" —
read `TODO.md` and reconcile it against the current state, not the
in-session task list alone. The in-session task list (`TaskList`) tracks
**active** work for this conversation; `TODO.md` is the durable
cross-session ledger.

**Update protocol:**
- When a task lands, move the line from §2 / §3 / §5 of `TODO.md` to §1
  (Completed work).
- Add a CHANGELOG.md entry, bump `CMakeLists.txt VERSION`, refresh
  README.md / CLAUDE.md if user-visible.
- Recurring task #33 (gap audit) re-walks `TODO.md` against the
  codebase periodically.

For exhaustive per-version implementation detail (parity gotchas,
forward-pass quirks, converter rules), the canonical sources are:
- `CHANGELOG.md` — every landing version's full root-cause-and-fix.
- `SESSION_DIGEST.md` — last session's per-version landing map.
- The relevant `src/models/yolo<N>.cpp` / `src/serialization/yolov<N>_weights.cpp` headers and the comments inside them.
- Memory keys (`~/.claude/projects/.../memory/`): `project_v6_parity_gotchas.md`, `project_v10_tf32_quirk.md`, `project_v13_arch.md`, `feedback_cli_auto_resolve.md`, `ref_full_matrix_sweep.md`, `project_state_0_24_0.md`.

## Periodic gap-audit (recurring TODO #33)

Re-run periodically (after major work lands, before declaring a phase
done, or whenever the user asks "what's left?"). The audit is not a
one-shot deliverable; it's a recurring sweep that keeps the documented
status honest.

**What to check, every time:**
1. **Per-version × task × pipeline matrix** — walk every cell of the
   capability matrix below. For each `—` cell, confirm it's still
   genuinely missing.
2. **Stub implementations** — grep for `not implemented yet` / `TODO`
   / `FIXME` / `XXX` / `unimplemented` / `unreachable` in `src/` and
   `include/`. Each hit is either tracked, a deferred follow-up to
   file, or stale (the code now handles it — delete the comment).
3. **Unwired CLI dispatches** — grep `cli/main.cpp` for
   `[error] ... not yet wired` blocks; each is a known gap with a
   concrete error path.
4. **SKIP-gated tests** — every `test_v*.cpp` that prints `SKIP` when
   weights/data are missing is a soft gap.
5. **Per-version variants** — confirm the (n,s,m,l,x + variant) table
   hasn't grown upstream without being mirrored.
6. **Parity status** — confirm any open `parity_compare` / per-version
   forward diff is still where it was last documented.

**What to produce:** a short report listing each gap with the tracking
task ID, scope estimate, and any blockers. Newly-discovered gaps go
into `TaskCreate`; closed gaps get `TaskUpdate complete`; stale matrix
cells get edits in the same pass.

**When to skip:** never automatically — but small one-off fixes (typo,
single comment) don't trigger it.

## Versioning + changelog policy (pre-1.0)

The project is **pre-1.0** until the user explicitly declares it
1.0-ready. Every release stays on `0.MINOR.PATCH`:

- **MINOR** bumps when a new YOLO version, task, or pipeline lands, or
  when the public surface changes in a non-additive way.
- **PATCH** bumps for additive changes (new test, new helper, bug fix,
  parity gotcha caught) that don't move the public surface.

**Single source of truth for the version literal:** the top-level
[`./VERSION`](VERSION) file (one line, `MAJOR.MINOR.PATCH`).
`CMakeLists.txt` `file(READ)`s it and feeds it to `project(yolocpp
VERSION ...)`, which CMake then exports into
`build/generated/yolocpp/config.hpp` as `YOLOCPP_VERSION_STRING`.
Consumers:

- `yolocpp --version` / `-v` / `-V` (and `yolocpp info`) print it at
  runtime.
- Future docs / release tooling read it through the same header — never
  duplicate.

**To bump the version, edit `./VERSION` only** (then add a
`CHANGELOG.md` heading). The only other legitimate places a literal
`X.Y.Z` should appear are:

1. `CHANGELOG.md` — `## [X.Y.Z] — YYYY-MM-DD` headings (immutable
   history).
2. Historical "landed in X.Y.Z" / "added 0.20.0" lines inside TODO.md
   tables — refer to a specific past commit, also immutable.

Anything else (README front-matter, prose snapshots, "current version
is …" lines) must read through the CMake-stamped string or be cut.
Task #47 owns this clean-up; #47A landed (the `./VERSION` file
itself), #47B/C remain.

**Every code change MUST be documented**:
1. Add a `## [X.Y.Z] — YYYY-MM-DD` heading at the **top** of
   `CHANGELOG.md` with `### Added` / `### Changed` / `### Fixed` /
   `### Deferred` subsections as appropriate.
2. Bump `project(yolocpp VERSION X.Y.Z)` in `CMakeLists.txt` to match.
3. Reference the change in README.md if user-visible (without
   re-stamping the version into prose — refer to the CHANGELOG entry).

A new model version is MINOR; wiring an existing pipeline to a
previously-unsupported model is MINOR; a parity bug fix or regression
test is PATCH; a behavior-preserving refactor is PATCH.

Going to `1.0.0` happens **only** when the user says so — never
preemptively bump.

## Commit + push policy

- Commit at every logical step (TODO update, CLAUDE update, README
  update, code change, test). Keep messages **brief and concrete** —
  one short subject line, no multi-paragraph body unless the diff is
  genuinely cross-cutting.
- Commits must be authored as the maintainer (the existing git
  `user.name` / `user.email`). **Do not** add `Co-Authored-By: Claude`
  or `Generated with Claude Code` footers — they're explicitly
  unwanted.
- **Never push.** The maintainer pushes by hand. Don't add `git push`
  to scripts or run it from automation either.
- New tasks discovered mid-implementation: file under the parent's
  `#NA`/`#NB` suffix in TODO.md when related + dependent, otherwise
  append to the end of the queue.

## Project goal

A pure-C++ replacement for Ultralytics. **No Python at runtime.**
LibTorch for training/eval, TensorRT for deployment, OpenCV for I/O.

### Supported YOLO versions (closed set)

Exactly twelve, with **no `v`** in any filename, identifier, namespace,
class name, comment, or doc string:

```
yolo3   yolo4   yolo5   yolo6   yolo7   yolo8
yolo9   yolo10  yolo11  yolo12  yolo13  yolo26
```

Anything outside this set (v1, v2, v14..v25, v27+) is **not supported
and not planned**. Follow the convention everywhere — even when
referencing legacy upstream URLs that publish as `yolov<N>...pt`. The
single legitimate place where strings differ from the canonical form
is `src/cli/resolve.cpp::upstream_basename`.

### Per-version capability matrix (current state — see CMakeLists.txt for the version stamp)

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

v6 variants: n / s / m / l + s/m/l/x_mbla + n6/s6/m6/l6 (12 total).
v7 variants: base / tiny / x + w6 / e6 / d6 / e6e (7 total). v9 has no
`x` upstream; v13 has no `m` upstream (iMoonLab fork).

### Task coverage matrix (predict/val/export/benchmark)

```
                detect   classify   segment   pose      obb
predict (CLI)   ✓        ✓          ✓         ✓         ✓
val (CLI)       ✓        ✓          ✓         ✓         ✓
ONNX export     ✓        ✓          ✓         ✓         ✓
TRT export      ✓        ✓          ✓         ✓         ✓
benchmark       ✓        gap        gap       gap       gap
```

Only v8 / v11 / v26 ship the full 5-task weight family upstream;
v12 / v13 ship detect-only (v12/v13 task heads scaffolded in code,
need to be trained on COCO ourselves — planned future session).
Benchmark for non-detect tasks is the only remaining task-coverage
gap.

### Reference smokes / sweeps

- **End-to-end ctest** (`ctest --test-dir build`): 31 tests, all green
  on the latest release. Per-version smokes are
  `tests/test_v<N>_e2e.cpp` / `test_v<N>_train.cpp`; SKIP-gated when
  weights/data missing.
- **Full matrix sweep** (`scripts/full_matrix_sweep.sh`): walks every
  applicable (version, variant, task, mode) cell. Last-known-good
  reading is `PASS=152 FAIL=0 SKIP=0` (predict 121, val 4, train 3,
  export 12, benchmark 12).
- **Predict-task sweep** (`scripts/task_predict_sweep.sh`): all 75
  (v8/v11/v26 × n/s/m/l/x × 5 tasks).
- **ONNX-export sweep** (`scripts/onnx_export_sweep.sh`): same 75.
- **ONNX numerical parity vs Python** (cls max|Δ| through
  onnxruntime CPU on `arange(N)/(N-1)` input): v12/v13 ≤ 1.8e-7 (fp32
  noise); v8/v11 detect ≤ 0.01, classify ≤ 1e-5, segment ≤ 0.002,
  pose ≤ 0.002, obb ≤ 0.004; v26 e2e NMS-free format isn't directly
  comparable.

### Cross-cutting parity rules (always-on; baked into source)

These bit one or more cells when violated; they're already wired but
preserve them through any forward/converter changes.

- **BN epsilon**:
  - Ultralytics yaml-built models (v8/v9/v10/v11/v12/v13/v26 detect /
    seg / pose / obb): `BatchNorm2d.eps = 1e-3`.
  - Ultralytics classify (`*-cls.pt`): PyTorch default `1e-5`.
    Threaded via the thread-local `BnEpsScope` in `yolo8.cpp`; pushed
    by each `Yolo*Classify` ctor.
  - Meituan v6 (every variant): saved `.pt` files have BN `eps=1e-3`
    (verified). Our `ConvBNReLUImpl` hardcodes 1e-3. Using PyTorch's
    1e-5 silently saturates v6l6 cls (root cause of #42).
- **v6 V6ActScope** (thread-local SiLU vs ReLU toggle): outer scope is
  `V6ActScope(is_l || is_mbla)` for the whole model, but the neck's
  structural convs (reduce_layer*, Bifusion*'s cv1/cv2/cv3, downsample*)
  must register inside a nested `V6ActScope force_relu(false)` block —
  upstream's RepBiFPANNeck/CSPRepBiFPANNeck hardcodes ReLU there
  regardless of training_mode. Already wired in NeckImpl/NeckP6Impl.
- **v10 TRT must clear `kTF32` on every scale** (n/s/m/b/l/x). The
  RepVGGDW 7×7-dwconv-with-bias stack accumulates enough TF32 mantissa
  loss to saturate cls. Already wired in both `cmd_export` and
  `engine::run_benchmark`.
- **CLI scale auto-resolve**: never default `scale_s` to a literal in
  any CLI11 layer. Default empty; resolve via
  `cli::scale_from_filename(weights)` at function entry. The "scale=n"
  default has bitten three times (#40 v10, v11 benchmark, v6 export).
- **Sigmoid-cls + DFL output contract**: every model's `forward_eval`
  returns `[B, 4+nc, A]` xyxy + sigmoided cls — drop-in for
  `inference::nms`. Holds across v3/v5/v8/v9/v10/v11/v12/v13/v26
  (anchor-free) and v4/v7 (anchor-based, with version-specific decode).
- **v11/v12/v13 m+l+x scale override**: Ultralytics' `parse_model`
  forces `c3k=True` (or v12 `residual=True/mlp_ratio=1.2`, or v13
  `dsc3k=True`) at width≥1.0 regardless of YAML. Replicated in each
  v11/v12/v13 yaml-walker.

### v11/v12/v13/v26-specific fields

The newer families introduced infrastructure all subsequent versions
reuse:

- `DWConvImpl`, `DWConvBlockImpl` (v11) — non-templated forward so
  Sequential can hold a Sequential without breaking AnyModule. Child
  names `"0"`/`"1"` match Ultralytics' state-dict layout.
- `DetectImpl::legacy` flag: `true` for v3/v5/v8/v9 (cv3 = Conv → Conv
  → Conv2d), `false` for v11+ (cv3 = DWConvBlock×2 → Conv2d).
- `infer_model_info` (in `cli/model_info.cpp`) disambiguates v8l vs
  v11m via the C2PSA marker
  (`model.{9,10}.m.0.attn.qkv.conv.weight`).
- v12 `AAttn` fused 3C qkv with k=7 pe (`conv.bias=true` — the only
  Conv in the codebase that does); v13 `V13AAttn` separate qk/v convs
  with k=5 pe. ONNX emitters split accordingly.
- v13 `HyperACE` parse_model rules: `c1 = ch[f[1]]` (second from-source);
  `num_hyperedges` scales 0.5/1.0/1.0/1.5 across n/s/l/x;
  `channel_adjust=True` for n/s, `False` for l/x; `DSC3k2`/
  `V13A2C2f` get `dsc3k=True` / `residual=True, mlp_ratio=1.5` overrides
  at l/x. Gamma init = `0.01 * ones(c2)` (NOT `ones(c2)` like v12).

### Output convention (current)

All four user-mode CLI commands write under `runs/<mode>/`:
- `runs/train/<exp>/...` (per-train-run dir, unchanged from earlier)
- `runs/predict/<source_stem>[_task].jpg`
- `runs/val/<weights_stem>_results.txt` (mAP, weights, data, imgsz)
- `runs/export/<base>.onnx` / `<base>.trt` (TRT cleans up its
  intermediate `.tmp.onnx`)

Caller's explicit `out=path` always overrides the default. Auto-created
via `std::filesystem::create_directories`.

### Out-of-scope (won't fix)

Documented as ❌ in `TODO.md §3`, NOT regressions:

- Original Darknet anchor-based v3 head (`.weights` binary loader).
- v6 lite / face variants.
- v7 `IAuxDetect` (training-only, stripped at deploy upstream).
- v9 PGI auxiliary branch (training-only, intentionally not wired).
- v3/v4/v5/v6/v7/v9/v10/v12/v13 cls/seg/pose/obb weights — upstream
  ships only detect; v8/v11/v26 are the full-task families.
  v12/v13 task heads scaffolded; need COCO-trained weights (future
  session).

## Build, test, run

```bash
./scripts/install_third_party.sh                         # one-time, ~5 GB
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure               # all tests
ctest --test-dir build -R smoke_test --output-on-failure # one test by regex
./build/yolocpp info                                     # CLI sanity check
bash scripts/full_matrix_sweep.sh                        # full version × mode matrix
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `89;90;120` and
`TORCH_CUDA_ARCH_LIST` to `8.9;9.0;12.0` — set those if targeting
different hardware. nvcc must be on PATH (or pass
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`).

## Toolchain — non-obvious choices

- **LibTorch is `cu130`**, not `cu128`. The system has CUDA 13.0 toolkit
  at `/usr/local/cuda` and PyTorch ships matching cu130 builds. Don't
  "downgrade" to cu128 unless you also reinstall the toolkit.
- **TensorRT install skips `libnvinfer-dev`** (the 2.9 GB deb — only
  static libs we don't link against). The unversioned `.so` symlinks
  the linker needs are produced by `scripts/install_third_party.sh`
  after extracting the runtime debs.
- **OpenCV is vendored from Ubuntu 24.04 universe debs**, not built
  from source. Runtime `.so.4.6.0` from `libopencv-*406t64`; headers +
  CMake configs from `libopencv-*-dev`. Both extracted under
  `third_party/opencv_root/`, exposed via `third_party/opencv/` symlinks.
- **Blackwell (sm_120) requires** TRT ≥ 10.13 with the cuda13.0 build.
  Smoke test only passes if
  `libnvinfer_builder_resource_sm120.so.10.14.1` is in
  `third_party/tensorrt/lib/`.

## The DT_RPATH gotcha — do not remove

Every executable uses
`target_link_options(... LINKER:--disable-new-dtags)` so the linker
emits **DT_RPATH** (transitive) instead of DT_RUNPATH (non-transitive).
Reason: `libnvinfer.so` calls `dlopen()` on its sm-specific resource
libraries (`libnvinfer_builder_resource_sm{75,80,86,89,90,100,120}.so.10.14.1`)
at engine-build time. With DT_RUNPATH the executable's rpath does NOT
propagate to those dlopen calls and you get:

```
IBuilder::buildSerializedNetwork: Error Code 6: API Usage Error
(Unable to load library: libnvinfer_builder_resource_sm120.so.10.14.1)
```

If you add new executables, give them the same `target_link_options`
and `BUILD_RPATH`/`INSTALL_RPATH` (use the `_yolocpp_rpath` variable
defined in the root CMakeLists).

## Per-version registry — how to add a new YOLO version

The CLI / public API dispatch on the version string ("v3", "v8", "v11",
…) through `yolocpp::registry::Registry`. Each supported version
registers a `VersionAdapter` (in
`include/yolocpp/registry/version_adapter.hpp`) describing the metadata
+ hooks the rest of the codebase needs. To add a new YOLO version:

1. **Drop the model implementation** into
   `src/models/yolo<N>.cpp` + `include/yolocpp/models/yolo<N>.hpp`
   following the existing convention (see yolo12 / yolo13 as recent
   templates).
2. **Add an emitter** in `src/serialization/onnx_export.cpp` exposing
   `export_yolo<N>_onnx(holder, path, cfg)`. Reuse helpers
   (`emit_conv_module`, `emit_bottleneck`, `emit_c3`, …) wherever the
   block primitives match.
3. **Register the version** by adding a `make_v<N>()` helper inside
   `src/registry/version_registry.cpp` that fills out a
   `VersionAdapter`, then add `r.register_version(make_v<N>());` to
   `register_all_versions()`. Per-version quirks (default imgsz,
   TF32-clear for TRT, supported task list) all live in the adapter.
4. **Wire the source file** into `yolocpp_core` in `CMakeLists.txt`.
5. **Add a smoke test** in `tests/test_v<N>_e2e.cpp` (SKIP-gated when
   weights are missing) and a registry assertion in
   `tests/test_registry.cpp` if the new version belongs to a
   distinguished class (e.g., full task family, anchor-based).

That's it — `cmd_export` (and, as #46D/E/F land, `cmd_predict` /
`cmd_val` / `cmd_train`) pick the adapter up automatically; no edits to
`cli/main.cpp`. As of writing #46A/B/C have landed (export migrated +
12-version registry seeded + walkthrough); predict/val/train are
tracked under #46D/#46E/#46F.

## Architecture commitments

Directory layout under `src/` and `include/yolocpp/` mirrors
Ultralytics: `models/`, `losses/`, `datasets/`, `engine/` (training
loop), `serialization/` (ONNX + TRT + .pt converters), `inference/`
(predictor + NMS + letterbox + TRT runtime), `core/` (device/version
utility), `cli/` (CLI11 + kv-style entry).

- **One unified training engine** (`TrainerT<M>` template + per-version
  `LossTraits<M>` specialization) across all model families.
- **One unified export pipeline** (`export_yolo<N>_onnx` + shared
  `build_trt_engine`) across all versions.
- Each task (detect / segment / pose / OBB / classify) = head + loss +
  dataset format + postproc; the rest is shared.
- Auto-resolve scale + version from filename via
  `cli::scale_from_filename` / `cli::version_from_filename` is the
  CLI's source of truth; never default to a literal scale letter.

## CLI surface

`yolocpp {train,val,predict,export,benchmark,info}` — all implemented.
Both kv-style (`yolocpp task=detect mode=predict model=...`) and CLI11
(`yolocpp predict --weights ...`) parsers are supported. Entry point
is `src/cli/main.cpp`.

## Editing third_party/

Files under `third_party/` are produced by
`scripts/install_third_party.sh` and are gitignored. **Don't commit
anything in there.** If a deb extraction needs to change (different
TRT version, different OpenCV mirror), update the script — not the
extracted tree.

## Parity validation

Numerical parity against Ultralytics is a hard requirement before any
model ships. Reference dumps come from a one-off Python tool kept
**outside the build** (dev-only, not in the runtime path) — uncommitted
venv at `/tmp/yolocpp_parity/.venv`, dumps under
`/tmp/yolocpp_parity/dumps/yolo<ver>/`. Per-layer parity tests live in
`tests/parity_compare.cpp` (registered in `tests/CMakeLists.txt` but
not always a ctest — gated on dump availability). v12/v13 add
`tests/test_v13_full.cpp` (cls max|Δ| ≤ 7.6e-10 across n/s/l/x) and
`tests/test_v13_ada.cpp` (six bit-exact module checks).

ONNX-vs-Python validation: `/tmp/yolocpp_parity/validate_onnx_tasks.py`
runs onnxruntime forward and compares per (version × task × scale).
