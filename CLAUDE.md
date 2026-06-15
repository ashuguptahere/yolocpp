# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with this repository.

## Engineering principles (read first, applies to every change)

These four rules override convenience. When they conflict with "just
make it work", they win.

- **Think before coding** — state your assumptions out loud, ask when
  unsure, never guess. Reading two files is cheaper than rewriting one.
- **Simplicity first** — write the minimum code that solves the
  problem, nothing extra. No premature abstractions, no
  "while-we're-here" refactors.
- **Surgical changes** — every changed line must trace back to the
  user's request. If you can't justify a hunk in a one-line PR
  comment, drop it from the diff.
- **Goal-driven** — turn vague instructions into verifiable success
  criteria before starting. "Make training faster" is not a goal;
  "AMP wired, cuDNN benchmark on, throughput ≥ 1.5× baseline on the
  v8n smoke" is.

### Language + style baseline

- **C++ standard: C++20** (`CMAKE_CXX_STANDARD 20`, CUDA on C++17). The
  codebase does not currently use any C++20-only feature; treat C++17
  as the de-facto floor and reach for C++20 features only when they
  earn their keep (`std::span` over `pointer+size`, `concept` over
  SFINAE if it shortens the call site, etc.).
- **RAII is non-negotiable.** No `new` / `delete` / `malloc` / `free`
  in `src/` or `include/`. Resources go through `std::unique_ptr` /
  `std::shared_ptr` / `std::ifstream` / `std::filesystem` / LibTorch
  refcounted tensors / the existing `BnEpsScope` / `V6ActScope`
  guard pattern. If you need a new resource type, wrap it in a
  custom-deleter `unique_ptr` — never raw owning pointers.
- **Don't reinvent STL / LibTorch.** Before writing a helper, check
  `<algorithm>`, `<numeric>`, `<ranges>`, `std::filesystem`,
  `torch::`, `at::`, and `cv::` first. Hand-rolled `clamp` / `min` /
  `sign` / `lerp` / string-split / file-read utilities are rejected
  on review. The legitimate exceptions are perf-critical kernels
  (NMS, letterbox) that exist for parity with upstream — those stay
  hand-written and are marked as such.
- **SOLID + KISS.** Single-responsibility per file; registry/adapter
  hooks for cross-cutting per-version behavior (see
  `version_adapter.hpp`). When a `.cpp` crosses ~800 lines, that's a
  smell — split by responsibility, not by line count.
- **Naming: `snake_case`** for functions / variables / namespaces;
  `PascalCase` for types / templates / enum values. This matches the
  existing codebase and LibTorch; it does NOT match Google C++ Style
  verbatim. If a tool enforces strict Google style, propose adding
  `.clang-format` / `.clang-tidy` before reformatting — don't do
  silent style sweeps.

### Build + run speed

The dev loop is the bottleneck before the model is. Keep these wired:

- **Generator: Ninja** when available (`cmake -S . -B build -G Ninja`).
  Falls back to Make if Ninja isn't installed, but Ninja's parallel
  scheduler is ~2× faster on cold rebuilds of this codebase.
- **Compiler launcher: ccache** when available. Wire via
  `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`. Single biggest dev-loop win
  on incremental rebuilds.
- **Linker: mold** (preferred) or **lld**. Pass
  `-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=mold` (or `lld`). GNU `gold` is
  the floor; default `ld.bfd` is too slow for a LibTorch link.
- **Builds are Release by default** (set in `CMakeLists.txt`). Don't
  override to Debug for "easier debugging" — it changes numerical
  behavior on tensor cores. Use `RelWithDebInfo` if you need symbols.

### Training speed (what fast actually means)

These knobs are mandatory in `TrainerT::run()` on CUDA, not optional:

1. **cuDNN benchmark on** — `at::globalContext().setBenchmarkCuDNN(true)`
   at trainer entry. Picks the fastest conv algo per shape.
2. **TF32 on** — `setAllowTF32CuBLAS(true)` + `setAllowTF32CuDNN(true)`.
   Ampere / Hopper / Blackwell tensor cores; ~1.3× MM throughput,
   numerically safe for training. (Inverse of the v10-TRT-export
   quirk — there we *clear* TF32; for training it stays on.)
3. **AMP via bf16 autocast** — `at::autocast::set_autocast_enabled
   (at::kCUDA, true)` + `set_autocast_dtype(at::kCUDA, at::kBFloat16)`
   around the forward + loss block. bf16 has fp32's range so no
   `GradScaler` is needed on Blackwell. **Never advertise `amp=true`
   in metadata without wiring it** — that lied for two minor
   versions before being caught.
4. **`channels_last` on CUDA** — convert model + inputs to
   `torch::MemoryFormat::ChannelsLast`. Another 10–20% on Tensor
   Cores for conv-heavy nets.
5. **`BatchPrefetcher` with N worker threads** (landed 0.94.0) — a
   producer/consumer queue of pre-built batches, sized `2 * workers`,
   each worker owning its own seeded `std::mt19937`. Default
   workers=4; --workers CLI flag overrides. Pipelines mosaic +
   perspective + decode + letterbox ahead of GPU compute. Built in
   preference to `torch::data::make_data_loader` because it didn't
   require refactoring `YoloDataset`'s interface — same effect,
   smaller diff. Don't claim workers > 0 in `args.yaml` if `cfg_.workers`
   is 0 — the metadata now reports the real value.
6. **No per-batch `.item<T>()` host syncs** — accumulate loss tensors
   on-device and `.item()` once per epoch. Per-step `.item()` calls
   force a CUDA sync and serialise the pipeline.

If you add a perf-relevant knob, prove the win with a before/after
on the v8n smoke (`ctest -R test_v8_train`) and record it in the
CHANGELOG entry. No unverified perf claims.

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
preemptively bump. **When 0.99 fills up, go to 0.100, 0.101, …
not 1.0.0.** The maintainer's explicit instruction: keep the
0.MINOR scheme rolling past 0.99 — there's no 1.0 cap on MINOR.
PATCH still resets per MINOR.

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
- **Test before every commit.** Build (`cmake --build build`), run the
  relevant ctest subset, and exercise the touched CLI / API path on a
  real artefact when one is available. Never `git add -A` and commit
  on the assumption that "it should work" — verify each diff first.
- **Rethink before staging.** Before `git add`, re-read `git diff` and
  remove anything that isn't required by the task: stray
  whitespace-only edits, trial debug prints, half-finished
  refactors, unrelated rewordings, files touched by accident. The
  commit must be the *minimum* diff that achieves the goal — no
  drive-by changes piggy-backing on the same commit.
- New tasks discovered mid-implementation: file under the parent's
  `#NA`/`#NB` suffix in TODO.md when related + dependent, otherwise
  append to the end of the queue.

## Project goal

A pure-C++ replacement for the upstream Python toolkit. **No Python at runtime.**
LibTorch for training/eval, TensorRT for deployment, OpenCV for I/O.

### Supported YOLO versions (closed set)

Exactly fourteen, with **no `v`** in any filename, identifier,
namespace, class name, comment, or doc string:

```
yolo1   yolo2   yolo3   yolo4   yolo5   yolo6   yolo7
yolo8   yolo9   yolo10  yolo11  yolo12  yolo13  yolo26
```

Anything outside this set (v14..v25, v27+) is **not supported and
not planned**. Follow the convention everywhere — even when
referencing legacy upstream URLs that publish as `yolov<N>...pt` /
`yolov<N>.weights`. The single legitimate place where strings differ
from the canonical form is `src/cli/resolve.cpp::upstream_basename`.

**yolo1 / yolo2** are Darknet-era models (Redmon 2016 / 2017,
respectively). v1 has no BN and uses two FC layers on top of a
24-conv backbone (`leaky 0.1` everywhere); v2 uses Darknet-19 + BN +
a `reorg` passthrough layer + `region` anchor head with 5 k-means-
clustered anchors. Both ship predict-only at the moment — train /
ONNX / TRT are tracked under tasks #66..#69.

**Canonical input form for v1/v2 is `.pt`**, same as every other
version. The `src/serialization/yolov{1,2}_weights.cpp` converters
exist to bootstrap a `.pt` from pjreddie's `.weights` binary when
needed, but the runtime never touches `.weights` directly. The
one-shot tool `tools/convert_weights` pre-populates `data/yolo{1,2,4}.pt`
from any locally-available `.weights` files; after that, the
codebase consumes `data/*.pt` exclusively (CLI, registry, tests,
predictors). `.weights` ingestion is a maintenance operation, not a
runtime one.

### Per-version capability matrix (current state — see CMakeLists.txt for the version stamp)

```
              arch     predict       val      train             ONNX/TRT export
yolo1         ✅       ✅            ✅       ✅                ✅
yolo2         ✅       ✅(+tiny)     ✅       ✅                ✅
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

DETR-family models (RF-DETR, RT-DETR) have been **removed from this
repo** and will live in a separate repository. Don't reintroduce
`rfdetr` / `rtdetr` / `detr` / `hungarian` source files, registry
adapters, CLI dispatch branches, or doc references here.

v6 variants: n / s / m / l + s/m/l/x_mbla + n6/s6/m6/l6 (12 total).
v7 variants: base / tiny / x + w6 / e6 / d6 / e6e (7 total). v9 has no
`x` upstream; v13 has no `m` upstream (iMoonLab fork).

### Task coverage matrix (predict/val/export/benchmark)

```
                detect   classify       segment      pose         obb
predict (CLI)   ✓        ✓              ✓            ✓            ✓
val (CLI)       ✓        ✓              ✓            ✓            ✓
ONNX export     ✓        ✓              ✓            ✓            ✓
TRT export      ✓        ✓              ✓            ✓            ✓
benchmark       ✓        ✓(PT+TRT)      ✓(PT+TRT)    ✓(PT+TRT)    ✓(PT+TRT)
```

Only v8 / v11 / v26 ship the full 5-task weight family upstream.
v12 / v13 ship detect-only upstream, but their task heads
(Yolo12/Yolo13 Segment/Pose/OBB/Classify) are now implemented and the
**full task pipeline is wired** — train + val through `cmd_train_task` /
`cmd_val_task`, predict through `cmd_predict_task`, and **ONNX + TRT
export through the registry `export_onnx` hook** (the v13 heads in
`yolo13_tasks.cpp` mirror the parity-tested v13 detect backbone; the
exporters reuse the version-agnostic task-head emitters + standalone
`walk_v1{2,3}_bb_neck` trunks). No upstream task weights exist, so these
are trained on COCO via the #60 `scripts/train_matrix.sh` harness.
Non-detect benchmark now reports PyTorch + TRT per-format rows for all
four tasks (TRT-backed model adapters reuse the `validate_*_t`
metrics): segment/pose/obb get fp16 + int8 with mAP (verified
TRT-fp16 ≈ PT); classify gets fp16 timing (int8/top1 need a full
ImageNet val, the same limitation the PT path has). The only remaining
benchmark gap is the ONNX format — gated on #70 (cv::dnn can't run the
decode subgraph).

Predict (CLI) covers image / dir / glob **and video / URL / webcam** for
all five tasks — the non-detect frame loop (`run_task_video` in
`cmd_predict_task`, 0.107.0) constructs the matching per-version
predictor and annotates each frame via the shared
`inference::draw_{segments,poses,obbs,classify}` helpers, writing
`runs/predict/<stem>_<task>.mp4`. Predict + ONNX/TRT export of v8/v11/v26
task variants route through the registry; v12/v13 task export routes
through the registry too (0.106.0).

### Reference smokes / sweeps

- **End-to-end ctest** (`ctest --test-dir build`): 46 tests, all green
  on the latest release. Per-version smokes are
  `tests/test_v<N>_e2e.cpp` / `test_v<N>_train.cpp`; SKIP-gated when
  weights/data missing. Task-head coverage: `test_v12_v13_task_export`
  (data-free v12/v13 task ONNX smoke), `test_task_draw` (the video-loop
  draw helpers), and `test_task_cross_backend_parity` (#53C — PT↔TRT
  numerical parity for seg/pose/obb/classify, 16 cells at relL2 ≤ 1e-3,
  gated behind `YOLOCPP_TRT_PARITY=1` since it builds 16 engines).
- **Full matrix sweep** (`scripts/full_matrix_sweep.sh`): walks every
  applicable (version, variant, task, mode) cell. Last-known-good
  reading is `PASS=164 FAIL=0 SKIP=0` (predict 121, val 4, train 3,
  export 12, benchmark 12, trt-roundtrip 12).
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
  - Upstream yaml-built models (v8/v9/v10/v11/v12/v13/v26 detect /
    seg / pose / obb): `BatchNorm2d.eps = 1e-3`.
  - Upstream classify (`*-cls.pt`): PyTorch default `1e-5`.
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
- **v10 AND v13 TRT must clear `kTF32` on every scale.** v10: the
  RepVGGDW 7×7-dwconv-with-bias stack accumulates enough TF32 mantissa
  loss to saturate cls. v13: the V13AAttn attention + DSConv + HyperACE
  accumulation loses enough TF32 mantissa to drop borderline detections
  (v13/s TRT-fp32 returned 3 dets vs PT's 5 with TF32 on; 5/5 with it
  cleared). Both set `trt_disable_tf32 = true` in their `VersionAdapter`,
  which `cmd_export` / `engine::run_benchmark` honor.
- **CLI scale auto-resolve**: never default `scale_s` to a literal in
  any CLI11 layer. Default empty; resolve via
  `cli::scale_from_filename(weights)` at function entry. The "scale=n"
  default has bitten three times (#40 v10, v11 benchmark, v6 export).
- **CLI nc auto-resolve**: never default `nc` to a literal (80) in any
  CLI11 layer — a literal 80 default is indistinguishable from an explicit
  `--nc 80`. Pass the `nc < 0` sentinel when `--nc` wasn't given
  (`main.cpp`: `app.count("--nc") ? nc : -1`; `PredictArgs`/`ExportArgs.nc`
  default `-1`). Predict resolves `nc < 0` → task default (80 detect/seg,
  1000 classify, 15 obb) once at `cmd_predict_task` entry; export recovers
  the class count from the checkpoint head (`infer_model_info().nc`) in
  `cmd_export`. The `(nc < 0 || nc == 80)` overload silently overrode an
  explicit `--nc 80` for classify/obb predict + export (fixed
  0.107.3–0.107.6). The one legitimate `nc == 80` is yolo2.cpp's
  COCO-vs-VOC anchor selector.
- **Sigmoid-cls + DFL output contract**: every model's `forward_eval`
  returns `[B, 4+nc, A]` xyxy + sigmoided cls — drop-in for
  `inference::nms`. Holds across v3/v5/v8/v9/v10/v11/v12/v13/v26
  (anchor-free) and v4/v7 (anchor-based, with version-specific decode).
- **v11/v12/v13 m+l+x scale override**: the upstream `parse_model`
  forces `c3k=True` (or v12 `residual=True/mlp_ratio=1.2`, or v13
  `dsc3k=True`) at width≥1.0 regardless of YAML. Replicated in each
  v11/v12/v13 yaml-walker.

### v11/v12/v13/v26-specific fields

The newer families introduced infrastructure all subsequent versions
reuse:

- `DWConvImpl`, `DWConvBlockImpl` (v11) — non-templated forward so
  Sequential can hold a Sequential without breaking AnyModule. Child
  names `"0"`/`"1"` match the upstream state-dict layout.
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
- v3/v4/v5/v6/v7/v9/v10/v12/v13 cls/seg/pose/obb *weights* — upstream
  ships only detect; v8/v11/v26 are the full-task families. v12/v13 task
  heads + the full pipeline (train/val/predict/export/parity) are wired;
  only the COCO-trained weights are deferred to #60 (compute-bound). The
  other versions (v3/v4/v5/v6/v7/v9/v10) have no task heads — detect-only.

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

That's it — `cmd_export`, `cmd_predict_task`, `cmd_val`, `cmd_train`,
and `engine::run_benchmark` all pick the adapter up automatically; no
edits to `cli/main.cpp` or `engine/benchmark.cpp` per added version.
As of writing #46A..F3 have all landed (export + predict + val +
train + benchmark migrated; kv-style dispatcher collapsed onto the
single-source command bodies; 12-version registry seeded; walkthrough
in this file). #46H (concept-based base) is recorded as won't-fix —
std::function-erased hooks read more cleanly and don't force every
version to implement every hook; see TODO.md #46H for the full
evaluation.

## Architecture commitments

Directory layout under `src/` and `include/yolocpp/` mirrors
the upstream Python layout: `models/`, `losses/`, `datasets/`, `engine/` (training
loop), `serialization/` (ONNX + TRT + .pt converters), `inference/`
(predictor + NMS + letterbox + TRT runtime), `core/` (device/version
utility), `cli/` (CLI11 + kv-style entry), `web/` (the optional
`yolocpp_web` console — server-side Clay→HTML UI + cpp-httplib backend
calling the public `YOLO` API; no LibTorch in the browser).

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

**Single canonical parser: flag-style with `--mode`** (see #51J / #51K).

```
yolocpp --mode <train|predict|val|export|benchmark|info|download> [flags...]
```

Every option sits at the top level. Common flags:

| short | long                  | scope          |
|-------|-----------------------|----------------|
| -m    | --model / --weights   | every mode     |
| -s    | --source              | predict, benchmark |
| -d    | --data                | train, val     |
| -o    | --out                 | predict, export |
| -D    | --device              | every mode     |
| -i    | --imgsz               | every mode     |
| -e    | --epochs              | train          |
| -b    | --batch               | train          |
| -n    | --nc                  | predict, export|
| -c    | --conf                | predict        |
| -f    | --format              | export         |
| -p    | --precision           | export         |
|       | --seed                | train          |
|       | --export-after-train  | train          |
|       | --dataset             | download       |
|       | --task                | predict, val, train, export (default: detect) |

Per-mode required-flag validation lives in `cmd_dispatch_flag_style`
in `src/cli/main.cpp`; error messages follow the form
`[error] --mode=<X> needs --<flag>`. The per-version dispatch within
each mode goes through the registry (see #46).

**Dataset format dispatch.** `--data` accepts five forms, all funnelled
through `cli::make_dataset` → `YoloDataset`'s pre-loaded ctor (so
trainer + validator stay typed on a single dataset class without
virtual dispatch):

| `--data` value                     | loader        | source                        |
|------------------------------------|---------------|-------------------------------|
| `coco/` (`images/<split>` + `labels/<split>`) | `YoloDataset` | existing YOLO layout |
| `dataset.csv` / `.tsv`             | `FlatDataset` | single-file format with `split` column |
| `instances.json`                   | `CocoDataset` | COCO 2017 schema |
| `VOC2012/` (`JPEGImages` + `Annotations` + `ImageSets/Main`) | `VocDataset` | Pascal VOC layout |
| `data.yaml` / `.yml`               | resolved      | upstream-style yaml; root is the resolved dir |

The dispatcher decides by extension (`.csv` / `.tsv` / `.json` /
`.yaml`) or by directory layout (`JPEGImages/Annotations` ⇒ VOC, else
YOLO).

**Task routing.** `--task` defaults to `detect` and accepts `detect |
classify | segment | pose | obb`. Detect routes through the registry
for every supported YOLO version; the four non-detect tasks route
through `cmd_val_task` / `cmd_train_task` (val + train) and through
the task-aware `adapter->export_onnx(... task ...)` hook (export). The
non-detect paths use the v8 task families (`Yolo8Classify`,
`Yolo8Segment`, `Yolo8Pose`, `Yolo8OBB`) — those are the
architectures whose task heads ship with weights upstream. Predict +
export of v11/v26 task variants additionally route through the
registry.

## Public C++ API (`#include <yolocpp/api.hpp>`)

Chainable Python-style API for embedding yolocpp in C++ apps:

```cpp
yolocpp::YOLO m("yolo11s.pt");
m.to("auto");
m.predict({.source = "bus.jpg"});                       // single image
m.predict({.source = "frames/", .out = "annotated/"});  // dir fan-out
m.predict({.source = "video.mp4"});                     // frame loop
m.val({.data = "coco/data.yaml"});
m.train({.data = "coco/data.yaml", .epochs = 100, .seed = 42,
         .export_after_train = "onnx"});
m.export_({.format = "onnx", .precision = "fp16"});
```

Every method routes through `yolocpp::cli::cmd_*` (in
`src/cli/commands.cpp`) — same dispatch the CLI uses. Adding a new
CLI flag means adding the matching field to the corresponding Args
struct in `api.hpp` and forwarding it in `api.cpp`. The cmd_*
functions live in the `yolocpp_core` static lib so both `main.cpp`
(the CLI driver) and `api.cpp` (the public API) link against them;
`cmd_dispatch_flag_style` is the sole exception — it stays in
`main.cpp` because it's the only CLI11 consumer.

The previous kv-style parser (`task=detect mode=predict ...`) and the
legacy subcommand-style parser (`yolocpp predict -m ... -s ...`) were
removed under #51K (maintainer's request: one canonical parser only).
Don't add them back without a maintainer green-light.

## Editing third_party/

Files under `third_party/` are produced by
`scripts/install_third_party.sh` and are gitignored. **Don't commit
anything in there.** If a deb extraction needs to change (different
TRT version, different OpenCV mirror), update the script — not the
extracted tree.

The closed dependency set + pinned versions live in
`third_party/DEPS.md`. `scripts/audit_deps.sh` enforces the whitelist
— any new `find_package` / `target_link_libraries` / `third_party/`
directory fails the audit unless it's documented in DEPS.md. Adding
a new dependency requires (a) a real need (boost, protobuf, fmt,
GTest are all on the explicitly-rejected list with reasons),
(b) Apache-2.0 / MIT / BSD-* compat (no GPL — see #50 reasoning),
and (c) a row in `DEPS.md` with version + license + size.

## Parity validation

Numerical parity against the upstream Python forward is a hard requirement before any
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
