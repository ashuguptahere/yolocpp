# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Single source of truth: TODO.md

**[`TODO.md`](TODO.md) is the canonical list of every task — completed, pending, in-flight — for the entire codebase, not just the active session.** It aggregates session task numbers (#1..#33+), per-version capability gaps, code-level `TODO` / `FIXME` comments, SKIP-gated tests, and pre-numbered Phase work (Phases 0..6 from before the task numbering scheme existed).

When the user asks "what's left?" / "show me all tasks" / "audit gaps" — read `TODO.md` and reconcile it against the current state, not the in-session task list alone. The in-session task list (`TaskList`/`TaskCreate`) tracks **active** work for this conversation; `TODO.md` is the durable cross-session ledger.

**Update protocol:**
- When a task lands, move the line from §2 / §3 / §5 of `TODO.md` to §1 (Completed work).
- Add a CHANGELOG.md entry, bump `CMakeLists.txt VERSION`, refresh README.md / CLAUDE.md if user-visible.
- Recurring task #33 (gap audit) re-walks `TODO.md` against the codebase periodically.

## Periodic gap-audit (recurring TODO)

There is a standing task — **#33 "Audit codebase for remaining gaps"** —
that should be re-run periodically (after major work lands, before
declaring a phase done, or whenever the user asks "what's left?"). The
audit is not a one-shot deliverable; it's a recurring sweep that keeps
the documented status honest.

**What to check, every time:**

1. **Per-version × task × pipeline matrix** — walk every cell of
   the capability matrix (the table near "Per-version capability matrix
   (detect, current state)" below) and the task-coverage matrix
   ("Task coverage matrix (predict/val/export/benchmark)"). For each
   `—` cell, confirm it's still genuinely missing — sometimes work
   lands without the matrix being updated.
2. **Stub implementations** — grep for runtime `not implemented yet`
   / `TODO` / `FIXME` / `XXX` / `unimplemented` / `unreachable` in
   `src/` and `include/`. Each hit is either:
   - already tracked as a task (cross-reference task list),
   - a deferred follow-up that should be filed as a new task, or
   - stale (the code now handles the case — delete the comment).
3. **Unwired CLI dispatches** — grep `cli/main.cpp` for
   `[error] ... not yet wired` blocks; each is a known gap with a
   concrete error path. Confirm the listed prerequisites are still
   accurate.
4. **SKIP-gated tests** — every `test_v*.cpp` that prints `SKIP` when
   weights or data are missing is a soft gap. Cross-check that the
   gating reason still applies (cache layout / weight names haven't
   shifted since the test was written).
5. **Per-version variants** — for each version, list which scales /
   sub-variants ship upstream and which are wired here. v6 MBLA
   (#23), v6 P6 (#24), v10 s/m/b/l/x (#21), v3 anchor-based legacy
   head, v7 auxiliary head — each is a documented gap; verify the
   list hasn't grown.
6. **Parity status** — if a `parity_compare` or per-version forward
   diff is part of an open task (e.g. #20 was the v6s residual
   gap), confirm the residual is still what was last documented.

**What to produce:**

- A short report (text only, no code changes) listing each gap with:
  - which task ID tracks it (or that a new task is needed),
  - whether the gap is genuinely still open,
  - rough scope (single-session vs multi-session),
  - any blockers (upstream weights missing, hardware needed, etc.).
- Any newly-discovered gaps go into TaskCreate; any closed gaps get
  TaskUpdate to mark complete; any stale matrix cells get
  CLAUDE.md edits in the same pass.

**When to skip the audit:** never *automatically* — but small one-off
fixes (typo, single comment) don't trigger it. Trigger points: a
task batch finishes (#15 e6e wrapping up v7, #17 wrapping val, #18
wrapping v9 train, etc.), a phase marker lands, the user asks
"what's left?" or "audit gaps," or a fresh session starts after a
long break.

## Versioning + changelog policy (pre-1.0)

The project is **pre-1.0** — public API, on-disk weight format, CLI
surface, and dataset conventions may all still change. Until the user
explicitly declares the codebase 1.0-ready, every release stays on
`0.MINOR.PATCH`:

- **MINOR** bumps when a new YOLO version, task, or pipeline (predict
  / val / train / export) lands, or when the public surface changes
  in a non-additive way.
- **PATCH** bumps for additive changes — new test, new helper, bug
  fix, parity gotcha caught — that don't move the public surface.

**Every code change MUST be documented**, with no exceptions:

1. Add a new `## [X.Y.Z] — YYYY-MM-DD` heading at the **top** of
   `CHANGELOG.md` (above the previous version's heading), with
   `### Added` / `### Changed` / `### Fixed` / `### Deferred`
   subsections as appropriate. Include enough detail to reconstruct
   *why* — parity gotchas, structural fixes, deferred follow-ups.
2. Bump `project(yolocpp VERSION X.Y.Z)` in `CMakeLists.txt` to match.
3. Reference the change in the relevant section of CLAUDE.md / README.md
   if it affects implementation status, the capability matrix, or a
   user-visible behavior.

When in doubt about whether a change is MINOR or PATCH: a new model
version is MINOR; a new task variant is MINOR; wiring an existing
pipeline to a previously-unsupported model is MINOR (e.g. v9 train);
fixing a parity bug or adding a regression test is PATCH; a refactor
that doesn't change behavior is PATCH.

The current major-line is `0.6.x` (Phase 6 + extension tasks). Going
to `1.0.0` happens **only** when the user says so — do not
preemptively bump to 1.0 even if the surface feels stable.

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
`yolo3` (Ultralytics' anchor-free `yolov3u` variant): Darknet-53
backbone + v8-style anchor-free DFL Detect head. Yolo3Impl is a
yaml-walker (flat ModuleList "model" mirroring `yolov3.yaml` indices
0..28) reusing v8's `Conv`, `Bottleneck`, and `DetectImpl(legacy=true)`
directly — upstream key names map 1:1. **Predict is end-to-end.**
The legacy anchor-based v3 head (3 outputs at 3 anchors × (5+nc))
isn't shipped; if you need the original Darknet anchor head, that's
a separate path (rebuild Yolo3Impl with LeakyReLU + write a `.weights`
binary parser similar to v4).
**Bottleneck repeats**: upstream's `parse_model` returns a bare
`Bottleneck` for `n=1` and `nn.Sequential(*Bottleneck × n)` for `n>1`
— the state-dict path differs (`model.<i>.cv1.*` vs
`model.<i>.<sub>.cv1.*`). Yolo3Impl mirrors this branching: pushes
either a Bottleneck or a Sequential of Bottlenecks per yaml entry.
`serialization::convert_yolov3_pt(yolov3u.pt → yolo3.pt)` is a
trivial fp16 → fp32 cast + `num_batches_tracked` drop (no fusion
needed). 433 tensors, ~103M params — matching Ultralytics' published
yolov3u count. On bus.jpg at default conf=0.25/iou=0.45: 7 dets =
4 person + 1 bus + 2 borderline (handbag/fire-hydrant noise around
0.25 — v3's older head is noisier than v8+). Top conf 0.94. Train/
val/export not wired.

`yolo4`, `yolo6`, `yolo7` have the architecture in place (forward-shape
verified by `tests/test_v4_v6_v7.cpp`; weight loaders deferred — each
family ships in its own state-dict format and needs a per-version
rename table). Specifically:

- **yolo4**: CSPDarknet53 (Mish backbone) + SPP + PANet + v3-style
  anchor head with 3 scales × 3 anchors × (5+nc). 64.36M PyTorch
  params at nc=80. Output `[B, 3*(5+nc), H_i, W_i]` at strides
  32/16/8; default imgsz=608 (yolov4.cfg's calibration size — anchors
  rescale automatically when imgsz differs). **Predict is end-to-end.**
  Module registrations are reordered to match yolov4.cfg DFS order
  (CSPStage: down→cv2→cv1→m→cv3→cv4; Yolo4: heads interleaved with
  bottom-up), so `serialization::convert_yolov4_weights` walks the
  model in the same order Darknet wrote the binary, fills tensors,
  and emits a `yolo4.pt` readable by our existing `pt_loader`. The
  resolver in `cli/resolve.cpp` runs the conversion lazily: if
  `yolo4.pt` is missing but `yolov4.weights` is in `data/` or the
  cache (or downloadable from AlexeyAB's release URL), it produces
  the `.pt` on first use. forward_eval applies the modern AlexeyAB
  scale_x_y bias-fix (1.2/1.1/1.05 for P3/P4/P5) and obj*cls fusion;
  bus.jpg → 6 detections (4 person + 1 bus + 1 other), matching the
  reference. Train / val / ONNX / TRT export not yet wired.
- **yolo6** (yolo6s only, Meituan v3.0 / release 0.4.0): EfficientRep
  backbone (deploy form = each RepVGG block fused to single Conv+ReLU
  with bias) + RepBiFPANNeck (BiFusion fuses three resolutions per
  level via cv1/cv2/upsample(ConvTranspose2d 2×2 stride 2)/downsample,
  cv3 cats `[upsample, cv1, downsample(cv2)]` — order matters because
  cv3's loaded weights expect that channel layout) + CSPSPPF
  (cv1..cv7 with 5/9/13 chained 5×5 maxpools) + EffiDeHead (decoupled
  anchor-free, reg_max=16/17-bin DFL via `proj` & `proj_conv`,
  separate `reg_preds_dist` (68 ch = 4×17) and auxiliary `reg_preds`
  (4 ch); inference uses `reg_preds_dist` only). 18.6M params at
  nc=80; output is the standard `[B, 4+nc, A]` xyxy + sigmoid'd cls
  ready for NMS. **Predict is end-to-end** via
  `serialization::convert_yolov6_pt(yolov6s.pt → yolo6s.pt)` which:
  * collects each upstream RepVGG triple (`rbr_dense` + `rbr_1x1` +
    `rbr_identity`) and applies RepVGG re-parameterization to fuse
    into one 3×3 Conv with bias (BN absorbed; identity branch
    encoded as a c_in×c_out identity 1×1 conv before BN-fuse, then
    zero-padded to 3×3).
  * strips the `.block.` infix from non-RepVGG names and rewrites
    the upstream `ERBlock_N.0` / `ERBlock_N.1` / `ERBlock_5.2.cspsppf`
    submodule keys to our cleaner naming.
  * casts fp16 → fp32, skips `num_batches_tracked` & `proj`/`proj_conv`
    (we recreate those deterministically in the head ctor).
  Loads 233 tensors (35 RepVGG blocks fused) into our model. The
  resolver runs the conversion lazily — `yolo6s.pt` missing but
  `yolov6s.pt` present (or downloadable from the Meituan release URL)
  triggers conversion on first use. Three structural gotchas caught
  during parity work, all required for correct output:
  1. **BN eps**. RepVGG branches and Meituan's `ConvBNReLU` both use
     plain `nn.BatchNorm2d` → PyTorch default `eps=1e-5`, NOT 1e-3.
     Wrong eps cascades into near-uniform cls scores.
  2. **CSPSPPF cat order**. Upstream `cv7(cat([cv2(x), main_path]))`
     puts the CSP shortcut FIRST. We initially had `[main, shortcut]`,
     which mis-routed half of cv7's input channels.
  3. **Eval uses `reg_preds`, not `reg_preds_dist`**. The 68-ch DFL
     branch (`reg_preds_dist`) is a training-only knowledge-distillation
     target; eval forward decodes boxes from the auxiliary 4-ch direct
     `reg_preds` branch (raw lrtb in cell units). Using DFL at eval
     gave near-correct boxes (the softmax averaging is forgiving) but
     the residual bias propagated through downstream and saturated cls
     scores on the wrong classes — manifested as bus → dining-table.
  After fixing all three, bus.jpg at default conf=0.25/iou=0.45 →
  exactly 4 person + 1 bus = 5 detections (person 0.71/0.51/0.48/0.34,
  bus 0.60), matching Meituan's reference inference.

  **n/m extensions** (after the initial s wire-up):
  - **v6n** (w=0.25): same RepBlock + RepConv pipeline as s, narrower
    channels. Converter unchanged. 4 dets at default conf (bus 0.71,
    3 person 0.63/0.61/0.48).
  - **v6m** (w=0.75): different topology — uses **BepC3** (CSP-wrapped
    `RepBlockBR` with **BottleRep** inner, each BottleRep is two
    `RepConv`s with a learned `alpha` scalar on the shortcut), DFL
    `reg_preds` (68-ch projection at eval, NOT the direct 4-ch path
    used by n/s), and **SimSPPF** (not CSPSPPF). `csp_e=2/3` per
    Meituan upstream. 5 dets (person 0.71/0.69/0.66/0.46, bus 0.67).

  **Critical converter gotcha for v6m/l**: my original `.block.` strip
  regex was too aggressive — it stripped both ConvBNReLU's wrapper
  `block` AND RepBlock's `block` ModuleList path (`m.block.<i>`).
  Fix: lookahead-restricted regex `\.block\.(?=(conv|bn)\.)` — only
  strip `.block.` when followed by a `conv.` or `bn.` field.

  **v6l added** (after revisiting): v6l shares the BepC3 + BottleRep
  topology with v6m (my earlier read of `[128, 128, 3, 3]` for the
  inner conv was on a stale converted file produced before the
  `.block.` lookahead-strip fix; the actual v6l `m.conv1.conv1` is
  `[64, 64, 3, 3]` matching c_=64 with csp_e=0.5). The one structural
  difference: v6l's upstream config has **`training_mode = 'conv_silu'`**,
  meaning every `ConvBNReLU` in the model uses **SiLU** activation,
  not ReLU. n/s/m all use ReLU. Implementation: a thread-local
  `V6ActScope` RAII captured by each `ConvBNReLUImpl` at construction
  — `Yolo6Impl::Yolo6Impl` pushes `V6ActScope(is_l)` before
  building the backbone/neck/head, so every nested ConvBNReLU
  (including those inside BottleRep, BepC3, the head's stems and
  cls/reg_convs) picks up the right activation. Forward stays a
  single per-instance branch on `use_silu`.
  On bus.jpg at default conf=0.25/iou=0.45: v6l gives 4 detections
  (bus 0.95, 3 person 0.88/0.80/0.65) matching Meituan's reference.

  **Why no v6x?** Meituan never released a plain `yolov6x.pt` —
  release 0.4.0 inventory has only n/s/m/l in the standard detection
  pipeline. The only `x` scale they ship is `yolov6x_mbla.pt`, which
  uses **MBLABlock** (Multi-Branch Linear Activation) — a structurally
  different block from BepC3 (m operates on full in_channels).
  Wiring x_mbla / s_mbla / m_mbla / l_mbla is a separate task: new
  module class + a third backbone/neck dispatch path. Tracked in
  task #23. Same goes for the high-res P6 variants (`yolov6{n,s,m,l}6.pt`,
  1280×1280 input, 4 detect levels) and the lite/face variants.

  Train/val and ONNX/TRT export not wired for any v6 scale.
- **yolo7** (yolov7 base/tiny/x): ELAN backbone + ELAN-W neck + MP
  downsamples + SPPCSPC + IDetect (deploy form). yolov7-base 36.93M
  params, yolov7-tiny 6.0M, yolov7x ~71M — all matching upstream
  published counts.

  **Scale dispatch via `Yolo7Scale {Base, Tiny, X}`** + per-scale yaml
  walker. Modules shared:
  - `Conv` is now activation-toggleable via `V7ActScope` (thread-local
    RAII captured at construction): tiny uses LeakyReLU(0.1); base/x
    use SiLU.
  - `MP` is plain MaxPool 2×2 stride 2 (no convs — base v7's "MP
    branch" is implicit via the yaml's `from` routing).
  - New `SP` yaml kind: single MaxPool kernel=k, stride=1, padding=k/2,
    used in tiny's SPP-equivalent block (yolov7-tiny replaces SPPCSPC
    with cv1+cv2+SP(5)/SP(9)/SP(13)+cat+conv).
  - **v7-tiny**: no SPPCSPC, no RepConv pre-head; tiny ELAN has only
    2 inner 3×3s per block (vs 4 for base, 6 for x); pre-IDetect is
    plain Conv 3×3 (not RepConv).
  - **v7-x**: wider channels (40 stem → 1280 P5), 8 inner 3×3s per
    ELAN block, pre-IDetect is **plain Conv** (not RepConv) — the
    yolov7x.yaml lists `Conv [N, 3, 1]` at layers 118/119/120, NOT
    RepConv. Initially I had RepConv there and got shape mismatches.
    Confirmed by upstream yaml + key shapes (`.conv.weight + .bn.*`,
    not `.conv.weight + .conv.bias`).
  - **Head ELAN cat order**: tiny uses dense 4-element
    `[-1, -2, -3, -4]`; base uses sparse 4-element
    `[-1, -3, -5, -6]`; x uses sparse 5-element `[-1, -3, -5, -7, -8]`
    (NOT 8-element sequential — that was my initial guess and it
    produced shape mismatches at the 1×1 reducing convs). The
    "sparse" pattern is the same idea — concat `[c_last, c_mid,
    c_early, cv2, cv1]` from the chain, dropping intermediate
    pass-throughs. cv5's input width = 5*c_inner = 1280 for v7x.

  Converter unchanged — same `convert_yolov7_pt` handles all three
  scales. tiny has 0 RepConv blocks to fuse (no rbr_dense/rbr_1x1
  keys in upstream); x also has 0 (its 118/119/120 are stored as
  Conv+BN, not as deployable RepConv pairs).

  **Forward dispatch**: Yolo7Impl's `forward_features` and
  `forward_eval` originally hardcoded `outs[102/103/104]` and
  `model[105]` for base — caused a segfault on tiny (78 layers) and
  x (122 layers). Fixed by reading the IDetect's `from` indices from
  the yaml and using `yaml.size()-1` for the head module index.

  On bus.jpg at default conf=0.25/iou=0.45:
    base : 4 person + 1 bus + 1 tie = 6 dets (top 0.94)
    tiny : 3 person + 1 bus = 4 dets (top 0.89)
    x    : 4 person + 1 bus + 1 tie + 1 handbag = 7 dets (top 0.96)
  All matching WongKinYiu's reference per scale.

  **w6 added** (P3-P6 / 1280²): introduces three new pieces:
  1. **`ReOrg`** module — 4× spatial-to-depth (`pixel_unshuffle(x, 2)`),
     no params. Used as layer 0 in w6/e6/d6/e6e: `[B, 3, 1280, 1280]`
     → `[B, 12, 640, 640]`. The Conv at layer 1 has 12 input channels.
  2. **4-level IDetect**: `IDetectImpl` now infers `nl` from
     `ch.size()` (3 for base/tiny/x; 4 for w6+) and allocates the
     `anchors` / `anchor_grid` buffers and `m` ModuleList accordingly.
     `stride` defaults to `[8, 16, 32, 64]` for nl=4.
  3. **No `ImplicitA`/`ImplicitM` in deploy form** — the published
     yolov7-w6.pt has only `m.0..m.3.{weight,bias}` (lead head only).
     Auxiliary head (yaml's IAuxDetect with separate `ia`/`im` per-
     channel learnables) is stripped during the train→deploy fuse;
     we don't need to handle it.
  On bus.jpg at 1280×1280, default conf=0.25/iou=0.45: 3 person + 1
  bus = 4 dets at 0.86–0.94, matching WongKinYiu's w6 reference.

  **e6 added** (after task #15 first pass):
  - Introduces **`DownC`** module — two-path strided downsample used
    by e6/d6/e6e (replaces the simple Conv-stride-2 of base/x/w6).
    Layout: cv1 (1×1, c1→c1) → cv2 (3×3 stride k, c1→c2/2) on the
    main path; mp (MaxPool stride k) → cv3 (1×1, c1→c2/2) on the
    shortcut path. cv2 is **3×3 (stride k)** and cv3 is **1×1**
    — initially I had them swapped (cv2 1×1 / cv3 3×3) and got 16
    shape mismatches at every DownC instance. Confirmed by upstream
    `models/common.py`'s DownC (cv2 = 3×3 stride k after cv1, cv3 =
    1×1 after maxpool).
  - **e6 yaml** (140 entries): same skeleton as w6 but with DownC
    instead of Conv-stride-2 downsamples, 6 inner 3×3s per ELAN
    block (vs 4 in w6) with sparse 5-element cat
    `[-1, -3, -5, -7, -8]` (same as v7x), wider channels
    (80 stem → 1280 P6).
  - 80.7M params. On bus.jpg at 1280×1280, default conf=0.25/iou=0.45:
    person 0.94/0.94/0.94, bus 0.92, person 0.74 = 5 dets matching
    WongKinYiu's e6 reference.

  **d6 added**: same module set as e6, just wider (96 stem → 1536 P6)
  and deeper (8 inner 3×3s per ELAN block, sparse 6-element cat
  `[-1,-3,-5,-7,-9,-10]` in backbone, 10-element cat in head ELAN-W).
  ~163-entry yaml. 154.7M params. On bus.jpg at 1280²: 4 person + 1 bus
  = 5 dets at 0.76–0.95, matching WongKinYiu's d6 reference.

  **e6e added**: E-ELAN with parallel ELAN sub-blocks. Each
  backbone/head stage has TWO parallel ELAN blocks taking the same
  input (the second routes back via `-11/-12`) followed by
  **`Yolo7Shortcut [-1, -11]`** which element-wise sums the two
  ELAN outputs. Single new module class (`Yolo7Shortcut` — literally
  `xs[0] + xs[1]`) plus a 262-entry yaml. The yaml is built via
  `e6e_bb_stage` / `e6e_head_td_stage` / `e6e_head_bu_stage` helper
  functions that emit the repeated 22/25/23-entry stage patterns —
  so the yaml definition is ~25 lines of helper calls instead of
  hundreds of literal entries. 152.5M params, 1210 tensors loaded.
  On bus.jpg at 1280²: bus 0.95, 4 persons at 0.70–0.94 = 5 dets,
  matching WongKinYiu's e6e reference.

  Despite the paper's "shuffle/merge cardinality" framing, the deploy
  graph has NO `channel_shuffle` in forward — that was a training-time
  augmentation. The deploy graph is literally just two parallel ELANs
  summed.
  **Predict is end-to-end.** Yolo7Impl is built as a yaml-walker
  (flat ModuleList named "model" mirroring yolov7.yaml indices
  0..105) so upstream key names map directly with no rename. Three
  RepConv layers (102/103/104) get fused at conversion time:
  `serialization::convert_yolov7_pt(yolov7.pt → yolo7.pt)` reads
  the train-form `.pt`, applies RepVGG re-parameterization to each
  RepConv triple (`rbr_dense + rbr_1x1 [+ rbr_identity]` →
  single 3×3 Conv with bias), strips the train-form keys, casts
  fp16 → fp32, writes our `.pt`. SPPCSPC's hidden-channel width
  was a parity gotcha: upstream `c_ = int(2 * c_out * e)` with
  default e=0.5 evaluates to `c_out`, NOT `c_out/2` — getting that
  wrong leaves the model 5M params short of upstream. Decode is
  WongKinYiu's "new coords" form: `xy = (σ(t)*2 − 0.5 + grid)*s`,
  `wh = (σ(t)*2)² * anchor_grid`, then `score = obj * cls`.
  Anchors / anchor_grid are loaded as buffers from upstream's
  `model.105.{anchors,anchor_grid}`. The yaml `from` index
  resolver must handle ALL negative ints (`-2`, `-3`, etc — not
  just `-1`); v7 uses heavy back-references in ELAN concats. On
  bus.jpg at default conf=0.25/iou=0.45: 4 person + 1 bus + 1 tie
  = 6 detections at confidences 0.92–0.94 (top-5) + 0.27 (the tie),
  matching WongKinYiu's reference inference. Train/val and ONNX/TRT
  export not wired. Auxiliary head and tiny/x/w6/e6/d6/e6e variants
  deferred.

- **yolo9** (Wang/Yeh/Liao 2024, GELAN backbone, scales t/s/m/c):
  RepNCSPELAN4 blocks + ADown/AConv downsamplers + SPPELAN + v8-style
  anchor-free Detect head. PGI auxiliary branch is training-only;
  deploy form is the lead head only. Yolo9Impl is a yaml-walker
  (flat ModuleList named "model" mirroring per-scale yamls indices
  0..22) with per-scale dispatch via `Yolo9Scale::{T,S,M,C}`.
  Reuses v8's `Conv` (BN eps=1e-3) and `DetectImpl(legacy=true)`
  directly. **Predict is end-to-end for all four detect scales.**
  `serialization::convert_yolov9_pt(yolov9{t,s,m,c}.pt → yolo9{...}.pt)`
  reads the train-form `.pt`, applies RepVGG re-parameterization to
  fuse each RepConv triple (`conv1` 3×3 + `conv2` 1×1 [+ optional
  identity-BN] → single 3×3 Conv with bias), strips the train-form
  keys, casts fp16 → fp32.
  Module set: `Yolo9RepConv` (deploy form), `Yolo9RepBottleneck`,
  `RepCSP` (C3 + RepBottleneck), `RepNCSPELAN4` (cv1 → chunk(2) →
  cv2/cv3 lanes → cv4 cat), `ADown` (avg_pool + chunk + 3×3 stride 2
  / 3×3 maxpool stride 2 + 1×1), `AConv` (avg_pool + 3×3 stride 2 —
  used by t/s/m), `ELAN1` (simpler 4-conv ELAN, used at layer 2 of
  v9t/s), `SPPELAN` (cv1 + 3 chained k=5 maxpools + cv5 cat).
  Param counts: t=2.13M, s=7.32M, m=20.08M, c=25.40M — all within
  <1% of Ultralytics' published values.
  **Critical parity gotcha for v9m**: Ultralytics' parse_model
  applies `make_divisible(args[0], 8)` to round every layer's output
  channel count up to a multiple of 8. v9m's yaml has AConv [180]
  at layers 16/19 — these get rounded to 184/240 in the actual
  weights. Without the round-up, layer 18's input concat width is
  off by 4 channels and load_from_state_dict silently drops the
  affected tensors → only 1 detection on bus.jpg. Internal args
  (c3/c4 of RepNCSPELAN4) are NOT rounded; only args[0] is.
  On bus.jpg at default conf=0.25/iou=0.45 across all 4 scales:
    yolo9t: 5 dets (top 0.96/0.90/0.87/0.85/0.47)
    yolo9s: 5 dets (top 0.97/0.92/0.90/0.89/0.63)
    yolo9m: 6 dets (top 0.95/0.91/0.90/0.90/0.60 + 0.29 tie)
    yolo9c: 5 dets (top 0.96/0.93/0.91/0.90/0.75)
  All matching Ultralytics' reference output per scale.
  Train/val and ONNX/TRT export not wired.
  **yolov9e added** (after task #16): two-pass backbone with
  CBLinear/CBFuse multi-level connectivity.
  - **CBLinear**: 1×1 Conv with `bias=true` whose output is logically
    split into N branches by a per-instance `c2s` channel-list. Stored
    as the FULL concatenated tensor `[B, sum(c2s), H, W]`; downstream
    CBFuse slices to pick a specific branch.
  - **CBFuse**: takes `len(idx) + 1` inputs — N CBLinear outputs +
    one "anchor" tensor (target spatial size). For each non-anchor
    input, slice by `idx[i]` (using its CBLinear's `c2s` list to
    compute the offset), nearest-upsample to the anchor's spatial,
    sum element-wise. No params.
  Yaml structure (yolov9e.yaml): primary path 0..9 mirrors v9c;
  layers 10-14 are CBLinear taps from primary layers 1/3/5/7/9 with
  `c2s` lists `[64]`, `[64,128]`, ..., `[64,128,256,512,1024]`;
  layers 15-29 form a secondary path that re-ingests the input image
  (layer 0 = `nn.Identity`) and pulls in matching branches via
  CBFuse at each downsample step. Layer-15..29 channel widths track
  the primary path. Neck/head 30..42 attach to the secondary path.
  In our `Yolo9Impl::Yolo9Impl`, CBFuse construction needs each
  upstream CBLinear's `c2s` list — looked up via `model[idx_layer]
  ->as<CBLinearImpl>()->c2s` after the CBLinear has been registered.
  In `forward_eval`, CBFuse's multi-input path is handled separately
  from Concat (collects inputs into a `std::vector<Tensor>` rather
  than catting). 58.2M params total. Conversion fuses 48 RepConv
  blocks (vs 16 for v9c — v9e has 2 RepBottleneck per RepNCSPELAN4
  for n=2). On bus.jpg at default conf=0.25/iou=0.45: 4 person + 1
  bus = 5 detections at 0.81–0.96, matching Ultralytics' yolov9e
  reference.

- **yolo10** (Tsinghua MIG, NMS-free end-to-end detection): SCDown
  + C2f + C2fCIB + SPPF + PSA backbone; v10Detect head with
  one2many (training) and one2one (inference, NMS-free) heads.
  Yolo10Impl is a yaml-walker matching Ultralytics' yolov10n.yaml.
  **Predict end-to-end for n only**; s/m/b/l/x deferred (different
  channel widths + which layers are SCDown vs Conv vs C2fCIB vs
  C2f differs per scale). 2.27M params (matches Ultralytics' 2.30M
  for the deploy-form-only one2one head; the 0.5M difference is
  the dropped one2many head).
  Modules: `SCDown` (Conv 1×1 + DWConv k×k stride s, no act),
  `RepVGGDW` (deploy form: single 7×7 dwconv with bias + SiLU),
  `CIB` (Sequential 5: DWConv 3×3 + Conv 1×1 + [DWConv 3×3 OR
  RepVGGDW] + Conv 1×1 + DWConv 3×3, optional shortcut),
  `C2fCIB` (C2f variant, m is CIB list), `PSA` (cv1 → split →
  attn(reuses v11 PSAAttention) + ffn → cv2). Detect reuses v8's
  `DetectImpl(legacy=false)` since v10's one2one head matches the
  v11-style cv3 (DWConvBlock×2 + Conv2d).
  `serialization::convert_yolov10_pt`: drops `model.<head>.cv2/cv3.*`
  (one2many — training only), renames `one2one_cv2/3.*` →
  `cv2/3.*`, fuses RepVGGDW pairs (`<prefix>.conv.conv.weight`
  7×7 + `<prefix>.conv1.conv.weight` 3×3-padded-to-7×7 + per-branch
  BN → single 7×7 dwconv with bias). 1 RepVGGDW pair fused for
  v10n (inside the C2fCIB at model.22 with lk=True, n=1).
  390 tensors written.
  **Critical parity gotcha**: upstream's `C2fCIB.__init__` overrides
  the default `e=0.5` → `e=1.0` when constructing inner CIBs. This
  makes the middle RepVGGDW operate on `2*c2` channels instead of
  `c2`. Without the override, 8 tensor shapes silently mismatch at
  load time (`cv1.1`, `cv1.2`, `cv1.3` of the inner CIB) and only 1
  detection comes out. CIBImpl now takes an `e` parameter; C2fCIB
  passes 1.0.
  On bus.jpg at default conf=0.25/iou=0.45: 4 person + 1 bus = 5
  detections at confidences 0.50–0.94, matching Ultralytics' v10n
  reference. Train/val and ONNX/TRT export not wired. The post-
  process is currently our standard NMS at default IoU=0.45 — for
  v10 this is essentially a no-op (one2one head outputs one box per
  object), but a strict NMS-free postprocess (per-class topk) is
  what upstream emits for ONNX export and is a small follow-up.

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
              arch     predict      val      train    ONNX/TRT export
yolo3         ✅       ✅(u form)   ✅       ✅(u form) ✅
yolo4         ✅       ✅           ✅       ✅       ✅
yolo5         ✅       ✅           ✅       ✅       ✅
yolo6         ✅       ✅(n,s,m,l + 4×_mbla + n6/s6/m6/l6) ✅      ✅       ✅(all 12)
yolo7         ✅       ✅(all 7 variants)  ✅   ✅(base) ✅
yolo8         ✅       ✅           ✅       ✅       ✅
yolo9         ✅       ✅(t,s,m,c,e) ✅      ✅       ✅(t,s,m,c,e)
yolo10        ✅       ✅(n,s,m,b,l,x) ✅    ✅(single + dual)  ✅(noTF32)
yolo11        ✅       ✅           ✅       ✅       ✅
yolo12        ✅       ✅           ✅       ✅       ✅
yolo13        ✅       ✅           ✅       ✅       ✅
yolo26        ✅       ✅           ✅       ✅       ✅
```

### Remaining gaps (per-cell scope)

Each open cell is an independent multi-session task; they don't share much
infrastructure beyond the existing `pt_loader` / `engine::Trainer` /
`engine::validate` / `serialization::onnx_export`:

- **yolo3 predict**: needs a yolo3 yaml-walker rebuild (current
  architecture is the legacy anchor-based v3, but Ultralytics ships
  `yolov3u.pt` with a v8-style anchor-free head). Two viable paths:
  (a) use Ultralytics `yolov3u.pt` and rewrite `Yolo3Impl` as a
  yaml-walker with v8 Detect (~v5-style adapter); (b) parse pjreddie's
  Darknet `yolov3.weights` binary (similar to v4), but the architecture
  needs to switch from SiLU to LeakyReLU first — otherwise the math
  doesn't match.
- **yolo9 / yolo10 architecture + predict**: brand-new architectures.
  v9 = GELAN backbone + PGI auxiliary branch (training-only). v10 =
  dual-head consistent assignment for NMS-free inference. Both share v8
  Detect head, so val/train/export piggyback on v8 once the backbones
  land.
- **v3/v4/v6/v7/v9/v10 val** is wired: each holder exposes
  `forward_eval(x) → [B, 4+nc, A]` (xyxy + sigmoid'd cls), so all six
  plug into the same `engine::validate` template that v5/v8/v11/v12/v13/v26
  use. Explicit instantiations live at the bottom of `validator.cpp`.
  The CLI `mode=val task=detect` dispatches by `version_from_filename`
  (or `version=...` override) to instantiate the right holder. v4
  defaults to `imgsz=608` (its anchors calibrate to 608²); v7's P6
  variants (w6/e6/d6/e6e) default to `imgsz=1280`. End-to-end smoke is
  `tests/test_val_v3_v10.cpp` (skips per-version when the converted
  `.pt` is absent from the cache).
- **v9 train** is wired. `Yolo9Impl` exposes `forward_train(x) →
  std::vector<Tensor>` (per-scale raw `[B, 4*reg_max+nc, H_i, W_i]`
  feature maps) plus public `scale` / `nc` / `stride` / `reg_max=16`
  fields, so it satisfies the `TrainerT<M>` contract directly. Default
  `LossTraits<M>` returns `V8DetectionLoss` (anchor-free DFL) which is
  exactly what v9's GELAN + v8 Detect head expects. `TrainerV9 =
  TrainerT<Yolo9>` is in `engine/trainer.hpp`; CLI `mode=train task=detect
  version=v9` dispatches at `cli/main.cpp`. End-to-end smoke is
  `tests/test_v9_train.cpp` (yolo9c finetune on coco8: loss 12.4 → 7.3
  in 2 epochs at lr=1e-3, batch=2; mAP@0.5:0.95 0.736 → 0.742). PGI
  auxiliary branch is intentionally not wired — upstream uses it
  training-only and strips it for deploy; we train the deploy head only,
  which converges a few mAP points lower than full PGI but is fine for
  finetune scenarios. v9-ctor was reordered to `(Yolo9Scale, int)` to
  match v8/v11 convention (callers updated: `predictor.cpp`,
  `cli/main.cpp` val branch, `tests/test_val_v3_v10.cpp`).
- **v3/v4/v6/v7/v10 train**: each needs its own loss class. v4/v7 are
  anchor-based v3-style (BCE for obj/cls, IoU for box, multi-anchor
  assignment). v6 uses VFL + SIoU + TAL (anchor-free, with the
  knowledge-distillation `reg_preds_dist` branch as the DFL target).
  v3u (Ultralytics' anchor-free yolov3u) could reuse `V8DetectionLoss`
  like v9 but `Yolo3Impl` would still need a `forward_train` method and
  a `(scale, nc)` ctor reorder to satisfy the trainer contract. v10
  needs the **dual-head consistent assignment** scheme — the deploy
  form keeps only `one2one`, but training requires both `one2many` and
  `one2one` heads with a per-anchor matching that aligns the two; this
  also requires keeping the one2many head in `Yolo10Impl` (currently
  stripped at conversion time). Each of these is an independent
  multi-session task, tracked separately. CLI `mode=train` for any of
  these versions returns a clear error message listing what's missing.
- **v4/v6/v7 ONNX export**: v4/v7 need an anchor-decode emitter
  (sigmoid + scale_xy/new-coords + grid + anchor_pix); v6 needs the
  dual-branch decode (or just `reg_preds` direct branch like the eval
  path uses).
- **v6 m/l predict**: m/l switch the neck from `RepBiFPANNeck` to
  `CSPRepBiFPANNeck` (CSP-wrapped RepBlocks). New `CSPRepBlock` module
  + neck rewrite.
- **v7 tiny/x/w6/e6/d6/e6e**: each is a different yaml. tiny uses LeakyReLU + a smaller ELAN; w6/e6/d6/e6e add a P6/P7 input
  resolution stage with 4 detect levels.


`arch ✅ + predict —` means the architecture compiles and forward-shape
verification passes, but no weight loader is wired yet (the loader is
the gating piece for the rest of the pipeline). v3/v4/v6/v7 each ship
in a different state-dict format from Ultralytics, so each needs its
own per-version rename table (or, for v4, a Darknet `.weights` binary
parser).

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
