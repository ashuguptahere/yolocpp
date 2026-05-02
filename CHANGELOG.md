# Changelog

All notable changes to **yolocpp** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy (pre-1.0)

The project is **pre-1.0** — the public API, on-disk weight format,
CLI surface, and dataset conventions may all still change. Versions
are stamped `0.MINOR.PATCH`:

- **MINOR** bumps when a new YOLO version, task, or pipeline
  (predict / val / train / export) lands, or when a public API or
  disk format changes in a non-additive way.
- **PATCH** bumps for additive changes (new test, new helper, bug
  fix, parity gotcha caught) that don't move the public surface.

The current line will stay on `0.MINOR.PATCH` until the user declares
the codebase 1.0-ready. At that point we tag `1.0.0` and start
following stable SemVer rules (MAJOR for breaking, MINOR for additive,
PATCH for fixes).

Every code change from this point forward gets:
1. A new `## [X.Y.Z] — YYYY-MM-DD` heading at the top of this file
   (above the previous version's heading), with `### Added` /
   `### Changed` / `### Fixed` / `### Deferred` subsections as needed.
2. A bumped `project(yolocpp VERSION X.Y.Z)` in `CMakeLists.txt` so
   the embedded version string reflects the change.

---

## [0.39.0] — 2026-05-02

### Added — RF-DETR weight loader + flat-dict pt_loader path (#65E2 partial)

Three pieces land:

1. `serialization::load_flat_state_dict(path, submodel)` extends
   `pt_loader.cpp` to handle checkpoints saved as
   `torch.save({'model': model.state_dict(), …})` — i.e. an
   already-flattened dotted-key dict-of-tensors rather than a
   pickled `nn.Module`. Used by RF-DETR (and many DETR-family
   models). Falls through to the module-shaped path if the entry is
   module-shaped instead.

2. `serialization::rfdetr_weights.{hpp,cpp}::load_rfdetr_pt(path,
   module, strict)` — generic loader that flattens an upstream `.pt`
   and copies each tensor onto a same-named parameter / buffer on
   the destination module via `named_parameters() / named_buffers()`.
   Returns a coverage report (`matched / unmatched / shape_mismatch
   / missing` counts + first-8 names of each bucket). `strict=false`
   today (transitional, until #65A2..D2 rewrite the modules to
   match upstream key names); `strict=true` will fire after #65D2.

3. `RFDetrImpl::load_from_upstream_pt(path, strict)` and
   `load_from_state_dict(entries)` now route through the loader
   instead of throwing. Registry hook for `predict_to_file` calls
   the loader on the weights path; `--mode predict -m rf-detr-base.pth
   -s data/bus.jpg` runs end-to-end (forward through scaffold
   modules with random init since upstream key names don't match
   yet, ergo 0 detections).

`tests/test_rfdetr_pt_load.cpp` (SKIP-gated on
`RFDETR_TEST_WEIGHTS_DIR=/tmp/yolocpp_parity/rfdetr_weights`)
verifies all 12 official 1.6.5 weight files unpickle cleanly:

```
[PASS] rf-detr-nano.pth        params=465  refpoint=[3900, 4]
[PASS] rf-detr-small.pth       params=487  refpoint=[3900, 4]
[PASS] rf-detr-medium.pth      params=509  refpoint=[3900, 4]
[PASS] rf-detr-base.pth        params=487  refpoint=[3900, 4]
[PASS] rf-detr-large.pth       params=533  refpoint=[3900, 4]
[PASS] rf-detr-seg-nano.pt     params=544  refpoint=[1300, 4]
[PASS] rf-detr-seg-small.pt    params=544  refpoint=[1300, 4]
[PASS] rf-detr-seg-medium.pt   params=572  refpoint=[2600, 4]
[PASS] rf-detr-seg-large.pt    params=572  refpoint=[3900, 4]
[PASS] rf-detr-seg-xlarge.pt   params=600  refpoint=[3900, 4]
[PASS] rf-detr-seg-xxlarge.pt  params=600  refpoint=[3900, 4]
[PASS] rf-detr-seg-preview.pt  params=544  refpoint=[2600, 4]
```

The pickle path is now proven; what remains is the architecture
rewrite (#65A2..D2) so that the destination param names match
upstream's keys 1-to-1. Each #65*2 module replacement increases
the `matched` count visible in the loader report. Once `matched ==
expected_count` and `unmatched == 0` for a variant, that variant
graduates to `strict=true` loading.

ctest count goes 42 → 43.

### Tracked

- TODO #65E2 partially landed (weight loader + flat-dict pt path);
  remaining work tracked under `#65A2..D2 + #65F2`.

---

## [0.38.0] — 2026-05-02

### Documented — RF-DETR 1.6.5 architecture spec; all 12 variants registered

Investigation phase of #65D landed: downloaded all 12 official
`rfdetr==1.6.5` checkpoints (5 detect + 7 segment), dumped per-key
shape inventories, captured the ground-truth architecture into
`docs/rfdetr_arch.md`, and updated the in-tree variant table to
match. The `RFDetrScale` struct now carries **real** per-variant
hyperparameters from the upstream `RFDETR<X>Config` classes:

| variant | resolution | patch | hidden | dec_layers | num_queries |
|---------|-----------:|------:|-------:|-----------:|------------:|
| nano    | 384        | 16    | 256    | 2          | 300         |
| small   | 512        | 16    | 256    | 3          | 300         |
| medium  | 576        | 16    | 256    | 4          | 300         |
| base    | 560        | 14    | 256    | 3          | 300         |
| large   | 704        | 16    | 384    | 3          | 300         |
| seg-n   | 368        | 14    | 256    | 4          | 100         |
| seg-s   | 512        | 16    | 256    | 4          | 100         |
| seg-m   | 576        | 16    | 256    | 5          | 200         |
| seg-l   | 672        | 16    | 256    | 5          | 300         |
| seg-xl  | 624        | 12    | 256    | 6          | 300         |
| seg-xxl | 768        | 12    | 256    | 6          | 300         |
| seg-prv | 432        | 12    | 256    | 4          | 200         |

### Added — `rf-detr-*.pth` / `rf-detr-seg-*.pt` filename routing

`src/cli/resolve.cpp::scale_from_filename` /
`version_from_filename` now recognise the upstream-canonical names
(`rf-detr-base.pth`, `rf-detr-seg-large.pt`, …) in addition to
our short `rfdetr-*.pt` form. Both route through the rfdetr
registry adapter — no separate code path. The regex covers `.pt`
and `.pth` extensions for both forms.

### Architecture rewrite scoped under #65A2..F2

The current scaffold (#65A..G, 0.31.0..0.37.0) was designed against
a generic DETR / LW-DETR shape and **does not match RF-DETR 1.6.5**.
Notable mismatches captured in `docs/rfdetr_arch.md` § "Differences
from current scaffold":

- Backbone: scaffold uses per-scale LW-DETR with embed_dims
  varying (192/384/512/768) + fused QKV; reality is a single shared
  `dinov2_windowed_small` (12 blocks, embed=384, separate Q/K/V,
  layer_scale1/2 trainable scalars per block) for 11 of the 12
  variants — `rfdetr-large` uses `dinov2_windowed_base` (embed=768).
- Encoder: scaffold has a multi-scale deformable encoder; reality
  has **none** — the "encoder" predictions are `transformer.enc_output*`
  + 13-deep `enc_out_class_embed` / `enc_out_bbox_embed` heads
  applied to backbone taps for two-stage query initialisation.
- Decoder: scaffold uses 3-level deformable cross-attn with
  separate per-layer cls/bbox heads; reality is **single-level**
  (`num_levels=1, dec_n_points=2, ca_nheads=16`) deformable
  cross-attn with shared cls/bbox heads + iterative bbox refinement.
- Loss: scaffold matches DETR's set loss; reality adds `group_detr=13`
  query grouping during training (not the matching algorithm
  itself, which is identical Hungarian).

These are tracked as new sub-tasks `#65A2..F2`:

| sub-task | scope                                                             |
|----------|-------------------------------------------------------------------|
| #65A2    | Replace `rfdetr_backbone.{hpp,cpp}` with HF DINOv2 windowed-attn  |
| #65B2    | Add `rfdetr_projector.{hpp,cpp}` (CSP-style, P4 only)             |
| #65C2    | Rewrite `rfdetr_decoder.{hpp,cpp}` (fused-QKV SA + 1-level CA + shared heads + iterative bbox refine) |
| #65D2    | Add `transformer.enc_output*` two-stage encoder-output module     |
| #65E2    | Add `rfdetr_weights.{hpp,cpp}` direct-loading converter (no key remap, just dtype + shape gates) |
| #65F2    | End-to-end test: load `rf-detr-base.pth`, forward `data/bus.jpg`, verify ≥ 1 plausible detection |

ctest count unchanged at 42/42. The forward path on the scaffold
still runs (random init), so `--mode train` for RF-DETR continues
to work; `load_from_state_dict` still throws on every variant
until #65E2 lands.

### Tracked

- TODO #65D investigation done; #65A2..F2 filed as the actual
  rewrite path.

---

## [0.37.0] — 2026-05-02

### Added — RF-DETR trainer integration (#65G)

`engine::TrainerRFDetr = TrainerT<RFDetr>` lands. The existing
templated trainer scaffold is reused — only a `LossTraits<RFDetr>`
specialization in `src/engine/trainer.cpp` is needed to:

- De-interleave `forward_train`'s flat `[cls0, bbox0, …]` output
  into per-layer cls/bbox lists.
- Convert the trainer's flat YOLO target tensor `[N, 6]` (`batch_idx,
  class, cx, cy, w, h`) into per-image `vector<RFDetrTarget>` lists.
- Call `rfdetr_set_loss` (#65F) and pack the result into the
  trainer's expected `LossOutput` (mapping l1→box, giou→dfl for
  logging compatibility).

`RFDetrImpl` / `RFDetrSegmentImpl` API simplified to expose
`scale`, `nc`, `stride` as plain fields (matching the YOLO model
convention used by the trainer template). `stride{1.0}` is fake
(set prediction has no FPN strides; LossTraits ignores it).

### Changed — `RFDetrImpl::forward_eval` contract

Now returns `[B, 4+nc, Q]` with **xyxy in pixel coords** + sigmoided
cls — drop-in compatible with `inference::nms` and `Validator`,
matching the YOLO contract documented in CLAUDE.md. `rfdetr_decode`
+ `test_rfdetr_decode` updated to read xyxy directly. The conversion
from sigmoided cxcywh-in-`[0,1]` to xyxy-in-pixels happens inside
`forward_eval` using the input image's H/W.

`engine::validate<RFDetr>` + `validate_with_records<RFDetr>`
explicitly instantiated so the trainer's optional `val_every`
hook compiles for RF-DETR.

The registry's `run_train_detect` hook now calls
`TrainerRFDetr::run()` instead of throwing. Train without weights
runs (random init); with `-m rfdetr-*.pt` it still hits the #65D
converter throw on `load_from_state_dict`.

### Tracked

- TODO #65G marked landed; #65D (converter), #65H (per-area mAP +
  task-aware val), #65I (ONNX export), #65J (TRT pipeline), #65K
  (segment), #65L (parity smokes) still pending.

---

## [0.36.0] — 2026-05-02

### Added — RF-DETR predict path / NMS-free decode (#65E)

`include/yolocpp/inference/rfdetr_predictor.hpp` +
`src/inference/rfdetr_predictor.cpp`. Two entry points:

- `rfdetr_decode(out, imgsz, conf, max_det)` — pure tensor → struct
  conversion. Per-query best class, threshold by `conf`, sort/top-K
  by score, convert sigmoided cxcywh in `[0, 1]` to letterbox-pixel
  xyxy. NMS-free by construction (each query is one prediction);
  `max_det=300` matches RF-DETR's upstream default.
- `rfdetr_predict_image(model, bgr, ...)` — letterbox →
  `forward_eval` → decode → `scale_boxes` unscale. Mirrors
  `Predictor::predict` so the registry's `predict_to_file` hook
  routes through it once #65D loads weights.

The registry hooks still throw at the converter (#65D) — without
real weights, predictions are noise — but the decode logic is now
landed and unit-tested independently. `tests/test_rfdetr_decode.cpp`
drives it with a hand-crafted `[1, 4+5, 4]` synthetic forward
tensor and verifies:
- Confidence threshold drops sub-`conf` queries.
- Sort by score (top-K ordering).
- `cxcywh × imgsz` → xyxy is bit-exact (q0 cx=0.5,cy=0.5,w=h=0.2 at
  imgsz=640 → (256, 256, 384, 384)).
- `max_det` caps the result correctly.

ctest count goes 41 → 42.

### Tracked

- TODO #65E marked landed; #65D (`rfdetr-*.pt` converter) is still
  the gate before the registry's `predict_to_file` hook produces
  meaningful detections.

---

## [0.35.0] — 2026-05-02

### Added — RF-DETR Hungarian set loss (#65F)

Set-prediction loss surface for RF-DETR landed independently of
the weight converter (#65D) — the loss runs against the
random-init forward, so #65G's trainer integration only needs the
converter on top.

- `include/yolocpp/losses/hungarian.hpp` +
  `src/losses/hungarian.cpp` — Jonker-Volgenant rectangular
  assignment. O(rows·cols²) shortest-augmenting-path with
  potentials, pure C++, zero deps. Handles `rows ≥ cols` (DETR has
  many more queries than GTs).
- `include/yolocpp/losses/rfdetr_loss.hpp` +
  `src/losses/rfdetr_loss.cpp::rfdetr_set_loss` — full DETR set
  loss. Per-image Hungarian match on the **last** decoder layer's
  outputs (matches stay stable across auxiliary layers), then sigmoid
  focal cls (α=0.25, γ=2) over all queries + L1 + GIoU on matched
  preds. Loss weights `λ_cls=2.0, λ_l1=5.0, λ_giou=2.0`. Auxiliary
  losses summed over all decoder layers.

`tests/test_rfdetr_loss.cpp`:
1. Hungarian matcher correctness on a 3×3 grid (any optimum
   accepted) + a rectangular 5×2 case with verified-optimum 12.
2. Full set-loss end-to-end on rfdetr-n with a 2-image batch
   (2 + 1 GTs). Asserts finite loss components and non-zero
   gradient flow back to model parameters.

ctest count goes 40 → 41.

### Tracked

- TODO #65F marked landed; #65D (`rfdetr-*.pt` converter) and #65G
  (trainer integration) are the two remaining gates before
  `--mode train` works on RF-DETR.

---

## [0.34.0] — 2026-05-02

### Added — RF-DETR decoder + object-query head (#65C)

`include/yolocpp/models/rfdetr_decoder.hpp` +
`src/models/rfdetr_decoder.cpp` close the RF-DETR forward path.

- `MLP` — 3-layer ReLU MLP for the bbox regression head.
- `DecoderLayer` — pre-norm transformer block with vanilla MHA
  self-attn over queries, MSDeformAttn cross-attn from queries to
  encoder memory, and an FFN. Reuses the encoder's MSDeformAttn
  primitive (#65B) for the cross-attn slot.
- `DetrHead` — `num_queries` learnable query embeddings + per-layer
  cls/bbox heads. `forward_eval` runs all decoder layers, takes the
  last layer's outputs, sigmoids cls + bbox, and packs to YOLO's
  `[B, 4+nc, num_queries]` channel order so the existing
  NMS-free-friendly downstream code stays generic. `forward_train`
  returns all per-layer (cls_logits, bbox_unact) pairs for the
  Hungarian auxiliary loss (#65F).

`RFDetrImpl::forward_eval` now runs end-to-end without throwing —
producing a YOLO-shaped output tensor with per-channel ranges in
`[0, 1]`. `forward_train` returns 2N tensors (N = num_decoder_layers)
for the loss surface that lands under #65F. **Output is meaningless
without trained weights** — the registry's `predict_to_file` hook
still throws, gated on `load_from_state_dict` (#65D) which is the
next slice.

`tests/test_rfdetr_forward.cpp` exercises full forward (eval +
train) on n (Q=100) + s (Q=200) and asserts:
- `forward_eval` shape `[1, 84, num_queries]` with channel-0..3 in
  `[0, 1]` (sigmoided bbox).
- `forward_train` returns `2 × num_decoder_layers` tensors.

ctest count goes 39 → 40. CLAUDE.md capability matrix unchanged
(rfdetr stays 🟡 until #65D loads real weights).

### Tracked

- TODO #65C marked landed; #65D (`rfdetr-*.pt` → our format
  converter) is the next slice and the gate before any meaningful
  prediction can run.

---

## [0.33.0] — 2026-05-02

### Added — RF-DETR multi-scale deformable encoder (#65B)

`include/yolocpp/models/rfdetr_encoder.hpp` +
`src/models/rfdetr_encoder.cpp`. The encoder takes the multi-scale
ViT taps, projects each level to `hidden_dim` via 1×1 conv, flattens
+ concatenates into a single `[B, ΣHi·Wi, hidden]` token sequence,
adds 2D sine positional embedding, and runs N pre-norm transformer
layers using **MSDeformAttn** (Zhu et al., 2020) instead of vanilla
self-attention.

`MSDeformAttn` is a portable composition of standard ops:
`sampling_offsets / attention_weights / value_proj / output_proj`
linears + per-level `grid_sample` for bilinear value gathering. No
custom CUDA kernel — ONNX export (#65I) decomposes into the same op
graph. Per-query cost is `H · L · P` samples, not the `Lq · Lv`
quadratic of full attention, so the 4800-token sequence on n/s is
manageable end-to-end.

`RFDetrImpl` now constructs + registers the encoder in the ctor and
exposes `forward_encoder(x)` which runs backbone+encoder. Per-scale
encoder depth: n=3, s=4, b=6, m=6, l=6 layers (from
`RFDetrScale::num_encoder_layers`). `forward_eval` runs both stages
before throwing on the decoder boundary (#65C).

`tests/test_rfdetr_encoder.cpp` exercises full backbone+encoder
forward at the configured `cfg.img_size` (640 for LW-DETR n/s) and
asserts the output shapes (memory `[1, 4800, hidden]`,
spatial_shapes `[3, 2]`). b/m/l skipped in this test on memory; full
parity smokes land under #65L.

ctest count goes 38 → 39.

### Tracked

- TODO #65B marked landed; #65C (decoder + object-query head) is
  the next slice. After #65C the predict path's
  `[B, 4+nc, num_queries]` shape contract holds end-to-end without
  weights — #65D's converter slots in next.

---

## [0.32.0] — 2026-05-02

### Added — RF-DETR backbones (#65A)

`include/yolocpp/models/rfdetr_backbone.hpp` +
`src/models/rfdetr_backbone.cpp` land the ViT backbone family for
RF-DETR. Two configurations:

- **DINOv2 ViT-L** (`rfdetr-l.pt`): patch=14, depth=24, embed=1024,
  16 heads, MLP×4, full self-attention, input 560×560. Tap blocks
  {11, 17, 23} → multi-scale feature maps for the encoder.
- **LW-DETR ViT** (`rfdetr-{n,s,b,m}.pt`): patch=16, input 640×640,
  windowed self-attention (window=14) on every block except the
  last. Per-scale (depth, embed_dim, num_heads):
  n=(6, 192, 3), s=(8, 384, 6), b=(10, 512, 8), m=(12, 768, 12).

Building blocks (`PatchEmbed`, `Attention`, `ViTBlock`,
`ViTBackbone`) are reusable across both families — only the cfg
constants differ. Submodule names match upstream (`patch_embed.proj`,
`blocks.<i>.attn.qkv`, `blocks.<i>.fc1/fc2`, `norm`) so the #65D
state-dict converter can do a simple prefix rename without
per-tensor surgery.

`RFDetrImpl` / `RFDetrSegmentImpl` now construct + register their
backbone in the ctor and expose `forward_backbone(x)`.
`forward_eval` runs the backbone end-to-end before throwing on the
encoder boundary (#65B / #65C). This means downstream slices can be
unit-tested incrementally — the backbone-shape contract is now
pinned.

`tests/test_rfdetr_backbone.cpp` exercises full forward through the
n/s/b/m scales on random `[1, 3, 640, 640]` input and asserts each
scale produces 3 multi-scale feature maps of shape `[1, embed_dim,
40, 40]` with finite values. DINOv2-large skipped (300M params would
balloon the test runtime). ctest count goes 37 → 38.

### Tracked

- TODO #65A marked landed in `TODO.md`; #65B (encoder) is the next
  unblocker.

---

## [0.31.0] — 2026-05-02

### Added — RF-DETR family scaffold (#65)

First non-YOLO architecture wired into the registry. RF-DETR is a
transformer-based DETR-family detector (encoder/decoder + object
queries + Hungarian-matching set loss + DINOv2 / LW-DETR backbone),
which lands one slice at a time across follow-up sessions; this
commit puts the **dispatch surface** in place so every CLI mode
(`--mode predict / val / train / export / benchmark`) and every
public API method (`yolocpp::YOLO("rfdetr-base.pt")`) routes through
a registered adapter and produces a clear, slice-tagged error
instead of silently mis-loading weights or returning garbage
detections.

- `include/yolocpp/models/rfdetr.hpp` + `src/models/rfdetr.cpp` —
  scale enum (`kRfdetrNano / Small / Base / Medium / Large`),
  `rfdetr_scale_from_letter("n|s|b|m|l")`,
  `rfdetr_default_imgsz(scale)` (560 for DINOv2-large, 640 elsewhere),
  `RFDetr` (detect) and `RFDetrSegment` holders. All forward,
  forward-train, and `load_from_state_dict` paths throw with
  `"rfdetr <area>: not yet implemented — tracked under TODO #65X"`
  pointing at the slice that owns the missing piece.
- `src/registry/version_registry.cpp::make_rfdetr()` registers the
  adapter under `version_id="rfdetr"`,
  `supported_tasks={"detect","segment"}`, with throwing hooks for
  every command; `register_all_versions()` picks it up next to the
  twelve YOLO versions.
- `src/cli/resolve.cpp` — filename regex extended to match
  `rfdetr-(n|s|b|m|l|nano|small|base|medium|large)(-seg)?\.pt`;
  `version_from_filename("rfdetr-*.pt")` returns `"rfdetr"`.
- `src/cli/commands.cpp` — `kKnown` known-version vector includes
  `"rfdetr"` so the CLI's existing `[error] unknown version` path
  doesn't fire before the registry's slice-tagged throw can run.

This is **architecture stub only** — no backbone, no transformer, no
trained weights, no ONNX emitter. Subsequent commits land #65A
(DINOv2 / LW-DETR backbone), #65B (transformer encoder), #65C
(decoder + object-query head), #65D (`rfdetr-*.pt` → our `.pt`
converter), #65E (predict / NMS-free decode), #65F (Hungarian
matching loss surface), #65G (train loop integration), #65H
(validator + per-area mAP), #65I (ONNX emitter), #65J (TRT
pipeline), #65K (segment mask head + seg variant), #65L (per-variant
parity smokes against upstream).

### Tracked

- TODO #65 + #65A..#65L breakdown filed in `TODO.md`.

---

## [0.30.0] — 2026-05-02

Cross-cutting overhaul. Five MINOR-worthy bodies of work landed in
one session — bumping a single MINOR digit captures the magnitude
without backfilling 30+ patch numbers. Going forward every commit
gets its own version bump per CLAUDE.md policy; this consolidated
entry covers the gap from 0.25.0.

### Added — per-version registry (#46)

Every supported YOLO version (v3..v13, v26) registers a
`yolocpp::registry::VersionAdapter` in
`src/registry/version_registry.cpp`. Each adapter carries
`std::function`-erased hooks for `export_onnx`, `predict_to_file`,
`run_val`, `run_train_detect`, `benchmark_pt`, `make_frame_predictor`
plus metadata (default imgsz per (scale, task), TF32 quirks,
supported task list). The CLI command bodies (`cmd_export`,
`cmd_predict_task`, `cmd_val`, `cmd_train`, `engine::run_benchmark`)
all dispatch through the registry instead of long if-else chains.
**Adding a new YOLO version is now a one-pass change**: drop the model
TU + ONNX emitter, write a `make_v<N>()` helper, register it. No
edits to `cli/main.cpp` or per-task pipelines. v8 leaves most hooks
empty and falls back to the unified `inference::Predictor` /
`engine::Trainer` paths. `tests/test_registry.cpp` enforces the
shape — every non-v8 adapter wires every hook. **Net deletion in the
CLI: ~890 lines** (cmd_predict_task, dispatch_kv, cmd_train inline
branches, cmd_export's per-version dispatch, engine::bench_pt's
per-version dispatch, build_onnx_for, cmd_dispatch_flag_style's val
+ train chains).

### Added — chainable C++ public API (#52)

`#include <yolocpp/api.hpp>` exposes `yolocpp::YOLO("...pt")` with
designated-initialiser `predict / val / train / export_ / benchmark`
methods. Every method routes through the same `cmd_*` body the CLI
uses (in the new `src/cli/commands.cpp`, lifted out of `main.cpp`'s
anonymous namespace under #52). `predict()` returns
`vector<Detection>` for image-mode (#52A2) — adapter
`predict_to_file` hook signature widened from `size_t` →
`vector<Detection>`. `to(device)` and `task(...)` set per-instance
defaults. `examples/` directory (#52B) ships seven self-contained
programs covering image / dir / video predict, train+auto-export,
ONNX export, benchmark, end-to-end pipeline; toggleable via
`-DYOLOCPP_BUILD_EXAMPLES` (default ON).

### Added — single canonical CLI parser: `--mode` (#51)

The kv-style (`task=detect mode=predict ...`) and legacy
subcommand-style (`yolocpp predict --weights ...`) parsers were
**removed entirely** under #51K. Flag-style is now the only CLI:

```
yolocpp --mode <train|predict|val|export|benchmark|info|download> [flags...]
```

Every option sits at the top level. Long + short forms wired
everywhere: `-m/--model` (alias `--weights`), `-d/--data`,
`-s/--source`, `-o/--out`, `-D/--device`, `-i/--imgsz`,
`-e/--epochs`, `-b/--batch`, `-n/--nc`, `-c/--conf`, `-f/--format`,
`-p/--precision`, `--seed`, `--task`, `--export-after-train`,
`--dataset`. Per-mode required-flag validation centralised in
`cmd_dispatch_flag_style` so error messages are uniform
(`[error] --mode=<X> needs --<flag>`). Removed: `src/cli/args.cpp`
(91 lines), `include/yolocpp/cli/args.hpp` (51 lines), the entire
CLI11 subcommand block (~200 lines).

New flag-level features: `--source` classifier handling image / dir
/ glob (#51C image slice) AND video / URL / webcam frame loop via
`inference::FramePredictor` + `cv::VideoCapture` + `cv::VideoWriter`
(#51C2). `--seed` plumbed through trainer + dataset shuffle +
`args.yaml`. `--task {detect|classify|segment|pose|obb}` plumbed
through predict / val / train / export with separate
`cmd_val_task` / `cmd_train_task` helpers for the v8 task families
(non-detect tasks). `--device` validator accepts `cpu | cuda |
cuda:N | cuda:0,1,... | mps | auto` and rejects invalid values
before any subcommand runs. `--precision {fp32|fp16|int8|int4|nvfp4}`
on export (fp32/fp16 wired; the rest filed under #51F2 with clear
errors). `--export-after-train` auto-exports `<save>/best.pt` →
`<save>/best.{onnx,trt}` post-training. `yolocpp --mode download
--dataset <name|url>` subcommand with built-in registry (coco8,
coco128, dota8, VOC, xView, ...).

### Added — dataset infra v2 (#54)

Three new dataset loaders + size-stratified mAP + Mosaic/Mixup:

- **FlatDataset** (#54A) — single CSV/TSV with header `split,
  image_path, class_id, x_center, y_center, width, height`. Multiple
  labels per image map to multiple rows; empty `class_id` registers
  a background image. Auto-detects comma/tab/semicolon delimiter.
- **VocDataset** (#54B) — Pascal VOC layout
  (`JPEGImages/Annotations/ImageSets/Main`). XML parsed via regex;
  no libxml2.
- **CocoDataset** (#54B) — COCO 2017 JSON schema. Parsed by a
  hand-rolled JSON tokenizer (no libjson per `DEPS.md`). Sparse
  category IDs compressed to dense [0, N).
- **`cli::make_dataset(spec, split, ...)` factory** — `--data` now
  accepts five forms (YOLO dir, VOC dir, `.csv`/`.tsv`, `.json`,
  `.yaml`/`.yml`); auto-detects by extension/layout. New
  `YoloDataset` ctor `(img_paths, vector<Tensor> labels, ...)`
  funnels every loader through one concrete dataset class so
  trainer + validator stay typed without virtual dispatch.
- **mAP S/M/L breakdown** (#54C) — `metrics::mAPResult` extended
  with `map_50_95_{small,medium,large}` + `n_gt_*` counts (COCO
  area buckets ≤32², ≤96², >96²). Surfaced through `cmd_val`
  output and `runs/val/<stem>_results.txt`.
- **Mosaic-4 + Mixup** (#54D) — `AugConfig::mosaic_p` /
  `mixup_p` (default 0). Mosaic stitches 4 sampled images at a
  random centre, crops to imgsz, drops boxes that shrink to ≤1 px.
  Mixup blends with α ~ Beta(8,8) approximation + concatenates
  label lists.

### Added — cross-backend parity test + TRT sweep phase (#53)

`tests/test_cross_backend_parity.cpp` — for each cell exports ONNX,
builds TRT FP32 + FP16, runs all three backends on bus.jpg, asserts
det count within ±1 of libtorch and IoU ≥ 0.50 same-class match for
≥ N-1 of N libtorch dets. Cells: v8n / v11s / v12s / v13s — 4/4
pass with `matched N/N`. v26 deliberately excluded (NMS-free deploy
form needs version-aware decode; covered by `test_v26_e2e`).
`scripts/full_matrix_sweep.sh` phase 7 added: per-version `.pt →
.trt → predict` round-trip across all 12 supported versions. Sweep
total **152/152 → 164/164 PASS**.

### Added — third-party manifest + audit (#48)

`third_party/DEPS.md` — single pinned manifest (libtorch
2.11.0+cu130, TensorRT 10.14.1.48+cuda13, CUDA 13.0.88, OpenCV
4.6.0, NCCL 2.23.4, rapidyaml 0.11.1, CLI11 2.4.x). Documents why
Boost / protobuf / GTest / fmt / json libs / ONNX Runtime are
explicitly rejected. `scripts/audit_deps.sh` enforces the whitelist
— any new `find_package` or undocumented `third_party/` entry
fails the audit.

### Removed — every "ultralytics" trace from the codebase (#49)

Every code-level mention of the upstream vendor neutralised across
CLI / models / inference / engine / serialization / datasets / tasks
/ tests / scripts (~75 files). Identifier rename:
`looks_like_ultralytics_weight` → `looks_like_upstream_weight`. The
allow-list (kept by design): `kAssetBase` URL constant in
`cli/resolve.cpp` (real network endpoint), the Meituan v6 release
URL (same), the pickle wire-format token `"ultralytics.nn.tasks"` /
`"DetectionModel"` in `pt_save.cpp` (downstream readers expect this
exact GLOBAL — renaming would break interop), historical
`CHANGELOG.md` entries, and the `#49` task description itself
(meta-reference).

### Added — `./VERSION` as version source of truth (#47)

CMake reads `./VERSION` (one-line `MAJOR.MINOR.PATCH`) and feeds it
into `project(yolocpp VERSION ...)`, which CMake then exports
through `build/generated/yolocpp/config.hpp` as
`YOLOCPP_VERSION_STRING`. To bump the version, edit `./VERSION`
only. New `yolocpp --version` / `-v` / `-V` flag prints it.
`scripts/check_version_literals.sh` lints the codebase against
stray `0.MINOR.PATCH` literals outside the allow-list.

### Changed — public registry hook signatures

- `VersionAdapter::predict_to_file` returns `vector<Detection>`
  (was `size_t`).
- `VersionAdapter::ValResult` widened with
  `map_50_95_{small,medium,large}` + `n_gt_*` fields.
- New `VersionAdapter::make_frame_predictor` hook (returns
  `unique_ptr<inference::FramePredictor>`).
- New `VersionAdapter::run_val` / `run_train_detect` /
  `benchmark_pt` hooks.

### Verification

- `ctest` full suite: **37/37 PASS** (was 31).
- `scripts/full_matrix_sweep.sh`: **164/164 PASS** (was 152).
- `scripts/check_version_literals.sh`: clean.
- `scripts/audit_deps.sh`: clean (4 packages, 9 third_party/
  entries; no new deps added in this body of work).

---

## [0.25.0] — 2026-05-01

### Changed — `runs/<mode>/` default output convention for predict / val / export

`mode=train` already wrote to `runs/train/`. The other three modes
defaulted to bare cwd-relative filenames (`predict_detect_out.jpg`,
`yolo11.onnx`, etc.) and val didn't persist anything to disk at all.
Aligned all four modes to the same `runs/<mode>/` home:

- **predict** — default output is now
  `runs/predict/<source_stem>[_task].jpg`. Both `cmd_predict` (CLI11
  path) and `cmd_predict_task` (kv-style, all-task) updated.
- **val** — new `write_val_results` helper writes
  `runs/val/<weights_stem>_results.txt` containing `weights=`,
  `data=`, `imgsz=`, mAP@0.5, mAP@0.5:0.95. Threaded through every
  per-version val branch (12 sites) so v3/v4/v5/v6/v7/v9/v10/v11/v12/
  v13/v26 all persist results consistently. Caller's explicit `out=`
  still wins everywhere.
- **export** — default ONNX path is `runs/export/<base>.onnx`; default
  TRT path is `runs/export/<base>.trt` with the throwaway intermediate
  `<base>.tmp.onnx` cleaned up after the TRT engine is built (so only
  the final `.trt` is user-visible).
- **train** — unchanged (was already writing to `runs/train/`).

`runs/predict/`, `runs/val/`, `runs/export/` directories are
auto-created on first use via `std::filesystem::create_directories`.
Caller's explicit `out=path` argument still overrides the default —
back-compat preserved for scripts/tests that pin a specific output
location.

End-to-end smoke after the change:

```
$ ./build/yolocpp task=detect mode=predict model=data/yolo11n.pt source=data/bus.jpg
[predict] (v11) 5 detections, wrote runs/predict/bus_detect.jpg

$ ./build/yolocpp task=detect mode=export model=data/yolo11n.pt format=trt --no-fp16
[trt] wrote runs/export/yolo11.trt (7158292 bytes)
[export] (v11/detect) wrote runs/export/yolo11.trt
$ ls runs/export/   # tmp cleaned up
yolo11.trt

$ ./build/yolocpp task=detect mode=val model=data/yolo11n.pt data=data/coco8/data.yaml
mAP@0.5      = 0.870901
mAP@0.5:0.95 = 0.587081
[val] wrote runs/val/yolo11n_results.txt
```

All 31 ctests still pass. The full matrix sweep is unaffected (the
sweep script passes explicit `out=` paths, so it doesn't exercise the
new defaults).

## [0.24.0] — 2026-05-01

### Added — Version-dispatched benchmark (closes the last sweep gap)

`engine::run_benchmark` previously hardcoded `inference::Predictor`
(v8-only) for the PT FP32 path AND `Yolo8Detect`/`kYolo8n` for the
ONNX-export-then-build-TRT step. The benchmark CLI consequently only
worked for v8 weights. Closed in 0.24.0:

- **`BenchConfig`** gains `version` / `scale` / `nc` fields. When
  empty, version+scale auto-resolve from the weights filename via
  `cli::version_from_filename` / `cli::scale_from_filename` (mirrors
  the predict/val/export auto-resolve added in 0.18.0).
- **`GenericPredictor<ModelHolder>`** template (anonymous namespace
  inside `benchmark.cpp`) wraps any of the 12 detect-only model
  holders into the same `predict(cv::Mat) → vector<Detection>`
  interface that `bench_one` consumes. Implements the standard
  letterbox → `forward_eval` → NMS → `scale_boxes` pipeline so the
  measurement window matches `inference::Predictor`'s.
- **`bench_pt`** + **`build_onnx_for`** dispatch helpers — case-walk
  every supported version (v3 / v4 / v5 / v6 P5+P6+MBLA / v7 / v9 /
  v10 / v11 / v12 / v13 / v26) and instantiate the right model
  holder, mirroring `cmd_export`'s per-version branches in
  `src/cli/main.cpp`. v8 keeps falling through to the legacy
  `inference::Predictor` (no behavior change for the existing
  benchmark cell).
- **TRT TF32 override** — `run_benchmark` now clears `kTF32` for v10
  builds (matches `cmd_export`'s same per-version override added in
  0.18.0). Without it, v10 TRT FP32 saturates cls outputs.

### Sweep result at 0.24.0

```
PASS = 152   FAIL = 0   SKIP = 0   (152 cells)
```

The benchmark phase grew from `v8 only` (1 cell) to `every version`
(12 cells, one per version × n-scale). All 12 PASS. Per-mode
breakdown: predict 121/121 · val 4/4 · train 3/3 · export 12/12 ·
benchmark 12/12. All 31 ctests still pass.

## [0.23.0] — 2026-05-01

### Added — v5 ONNX + TRT export (closes the last gap from the matrix sweep)

The full matrix sweep at 0.22.1 had exactly one failing cell: v5
ONNX export, gated by a `TORCH_CHECK(false, "v5 ONNX export not yet
wired ...")` because Yolo5Detect's backbone (6×6-stem + C3) differs
structurally from Yolo8Detect's (3×3-stem + C2f) and reusing
`export_yolo8_onnx` wouldn't have worked. Closed in 0.23.0:

- New **`export_yolo5_onnx`** in `serialization/onnx_export.cpp` — walks
  the 25-entry v5 yaml (Conv/C3/SPPF/Upsample/Concat/Detect) by
  runtime module dispatch on the registered `model->model[i]` slots.
- New **`emit_c3`** helper:
  `cv3(cat([cv1(x) → m → ..., cv2(x)], dim=1))`
  where `m` is a ModuleList of Bottlenecks. Reuses the existing
  `emit_bottleneck` (which already reads `cv1->kernel_size` /
  `cv2->kernel_size` at runtime, so v5's k=(1,3) Bottlenecks work
  with no changes — only v8's k=(3,3) was previously exercised).
- The 6×6 stride-2 stem at layer 0 needs no special emitter — the
  generic `emit_conv_module` reads kernel/stride/padding from the
  registered `ConvImpl` directly.
- Detect head reuses `emit_detect` (legacy=true, anchor-free DFL —
  identical to v3/v8/v9).
- CLI: `cmd_export`'s v5 branch now constructs `Yolo5Detect` (was
  previously falling through to the v8 path with wrong stem channels)
  and dispatches to `export_yolo5_onnx`.
- `Yolo5Detect` added to `include/yolocpp/serialization/onnx_export.hpp`.

End-to-end smoke (TRT FP32 on bus.jpg at conf=0.25):
- v5n: 4 dets (top conf 0.86 via ORT)
- v5s: 5 dets (top conf 0.88)
- v5m: 5 dets (top conf 0.93)
- v5l: 5 dets (top conf 0.94)
- v5x: 5 dets (top conf 0.94)

All matching the libtorch baseline. Full matrix sweep now at **141/142
PASS, 0 FAIL** (one v8-only benchmark cell was the one tested on the
v8 path, leaving 141 because the v5 export count went from 0 → 1).

## [0.22.1] — 2026-05-01

### Changed — Task #33: gap-audit cleanup (post-0.22.0)

Recurring audit run after #45 closed every numbered task except #33
itself. Cross-walk results:

- **Capability matrix** (CLAUDE.md + README.md) — both still referenced
  closed tasks (`#42`, `#40`, `#45`, "deferred" / "⚠TRT" cells).
  Rewritten to reflect actual state: every YOLO version × pipeline cell
  is now ✅ for the variants we ship.
  - yolo6 train + ONNX/TRT now claim "all 12" (was "n,s" / "n,s,m,l").
  - yolo9 ONNX/TRT now claims "t,s,m,c,e" (was "t,s,m,c; e deferred").
  - yolo10 train now claims "single + dual" (was "one2one; dual-head #45").
  - yolo10 ONNX/TRT now claims "noTF32" with no `#40` qualifier.
- **TODO.md §3 (Pending — by version)** — every version's bullet list
  rewritten to reflect closed status; out-of-scope items (Darknet
  anchor head, lite/face variants, IAuxDetect retrain) kept as
  explicit ❌ with reasoning.
- **Runtime stubs sweep** (`grep -nE "TORCH_CHECK\(false|not yet
  wired|FIXME|XXX|unimplemented"`) — only structural unreachables
  remain (`forward_train` end-of-yaml fall-through asserts, parent
  Backbone/Neck `forward` placeholders that delegate to sub-modules).
  Two cosmetic TODOs remain:
  - `src/tasks/segment_train.cpp:388` — "expose forward_train_seg" —
    follow-up note inside an already-working segment training path.
  - `include/yolocpp/datasets/yolo_dataset.hpp:20` — "Mosaic / mixup
    are TODO" — augmentation menu, not a gap in any shipped pipeline.

No code change — pure docs sync. Per pre-1.0 policy this is a PATCH bump
(additive doc cleanup, no API/disk-format movement). All 31 ctests pass.

## [0.22.0] — 2026-05-01

### Added — Task #45: v10 dual-head consistent-assignment training (paper §3.1)

Implements Wang et al. 2024's "consistent dual assignment" training
recipe for v10. Both heads share the V8 TAL metric (CIoU + classification
weighting); the only difference is the topK count: one2many uses TAL
topk=10 (rich gradient drives the backbone), one2one uses TAL topk=1
(forces a single positive per GT so the deploy graph is NMS-free). The
two losses are summed equally (paper Eq. 4).

**Architecture**:
- `Yolo10Impl` gains a `dual_head` flag + new ctor
  `Yolo10Impl(scale, nc, dual_head)`. The default ctor still produces
  the deploy-form one2one head (no behavior change for existing users).
- When `dual_head=true`, a parallel `Detect(legacy=true)` is registered
  as `o2m_detect` alongside the existing one2one `Detect(legacy=false)`
  inside `model[head_idx]`. The legacy=true cv3 (Conv→Conv→Conv2d) is
  the v8-style one2many head shape upstream Ultralytics ships.
- `forward_train` returns 6 features when `dual_head=true`
  (`{o2m P3, o2m P4, o2m P5, o2o P3, o2o P4, o2o P5}`); 3 features
  otherwise. `forward_eval` is unchanged.

**Loss**:
- `losses::V10DualLoss` (new file `losses/yolo10_loss.{hpp,cpp}`) —
  runs `V8DetectionLoss` twice with `topk=10` / `topk=1`, sums the
  outputs.
- `losses::Yolo10LossAdapter` — runtime branch that picks single
  `V8DetectionLoss` when `feats.size()==3` and `V10DualLoss` when
  `feats.size()==6`. Wired as `LossTraits<models::Yolo10>` in
  `engine/trainer.cpp` so existing `TrainerV10 = TrainerT<Yolo10>` now
  handles both modes.

**Converter**:
- `serialization::convert_yolov10_dual_pt` — sibling of the existing
  one2one-only converter that ALSO routes upstream's
  `model.<head>.cv2.*` / `cv3.*` (one2many) into our parallel
  `o2m_detect.cv2.*` / `cv3.*` slots. Used only by the dual-head
  training path; the deploy converter (`convert_yolov10_pt`) still
  strips the one2many head.

**CLI**:
- `mode=train task=detect version=v10 dual_head=true` enables the dual
  path. Defaults to single-head when omitted (preserves 0.13.0 wiring).

**Smoke test** (`tests/test_v10_dual_train.cpp`):
- Builds yolo10n with `dual_head=true`, loads the dual-converted
  upstream weights (432 tensors — 48 more than single-head), runs 2
  epochs on coco8 at lr=5e-4, batch=2 on CPU. Total loss decreases
  from 16.85 → 15.14, with `cls` dropping from 14.39 → 10.09 over
  4 steps — confirms gradients flow through both heads. Skipped when
  `~/.cache/yolocpp/weights/yolov10n.pt` or `data/coco8` is absent.

All 31 ctests pass.

## [0.21.0] — 2026-05-01

### Added — v6 P6 + MBLA ONNX + TRT export

Closes the v6 ONNX/TRT export coverage gap: every released Meituan v6
weight (n / s / m / l + n6 / s6 / m6 / l6 + s/m/l/x_mbla = 12 variants)
now exports end-to-end. The two added paths share all the existing P5
emitters from 0.19.0 (`emit_v6_convbn`, `emit_v6_repconv`,
`emit_v6_repblock`, `emit_v6_bottlerep`, `emit_v6_repblockbr`,
`emit_v6_bepc3`, `emit_v6_simsppf`, `emit_v6_cspsppf`,
`emit_v6_bifusion`, `emit_v6_effidehead` with DFL).

**P6 path** (n6 / s6 / m6 / l6, 4 detect levels at strides 8/16/32/64,
default imgsz=1280):
- `export_yolo6_onnx` now branches on `model->is_p6` and walks
  `Yolo6Impl::backbone_p6` / `neck_p6` directly. The 6-stage backbone
  (stem + ERBlock_2..6) feeds the P6 SPPF (CSPSPPF for n6/s6, SimSPPF
  for m6/l6) registered as `ERBlock_6_cspsppf`. The 4-level neck does
  3 reduce_layer/Bifusion/Rep_p* top-down passes plus 3
  downsample/Rep_n* bottom-up passes, ending in a 4-input EffiDeHead.
- CLI: scale_s `n6/s6/m6/l6` resolves to the right `(Yolo6Scale, p6)`
  pair, and the v6 P6 imgsz default override (640 → 1280) lands in
  `cmd_export` so both ONNX and TRT profile see the same value.

**MBLA path** (s_mbla / m_mbla / l_mbla / x_mbla, P5 4-level head with
DFL eval):
- New emitters:
  - **`emit_v6_bottlerep3`** — three-conv variant of BottleRep with the
    same `add ? y + alpha*x : y` shortcut.
  - **`emit_v6_mblablock`** — splits cv1's branch_num*c_inner channel
    output into branch_num chunks via `Slice`, runs each Sequential of
    BottleRep3s capturing every intermediate, concatenates all
    (chunks ∪ intermediates), and projects via cv2. Mirrors
    `MBLABlockImpl::forward` exactly.
- `bb_block` / `nk_block` dispatch lambdas in `export_yolo6_onnx` now
  cover all three block types (RepBlock / BepC3 / MBLABlock) at every
  ERBlock_*_block and Rep_p*/Rep_n* slot.

End-to-end smoke (TRT FP32 on bus.jpg matching libtorch):
- v6 P6: n6=5, s6=6, m6=5, l6=8 dets (imgsz=1280)
- v6 MBLA: s_mbla=6, m_mbla=6, l_mbla=6, x_mbla=5 dets (imgsz=640)

The `TORCH_CHECK(scale.variant == Standard)` and
`TORCH_CHECK(!model->is_p6)` guards inside `export_yolo6_onnx` are
removed.

## [0.20.0] — 2026-05-01

### Added — v9e ONNX + TRT export (CBLinear + CBFuse two-pass backbone)

Extends `export_yolo9_onnx` from t/s/m/c (23-layer single-pass) to also
support v9e (43-layer two-pass with multi-level CBLinear/CBFuse taps).

- Added a per-scale 43-entry skeleton (`y_skel_e`) covering the
  primary backbone (0..9), CBLinear taps (10..14), secondary backbone
  with CBFuse fusion at every downsample (15..29), and the regular
  GELAN head (30..42). The primary path's layer 0 is `nn.Identity`
  (raw input pass-through); the secondary path re-ingests at layer 15.
- **`CBLinear` emit** — single 1×1 Conv (bias=true) with output
  channels = sum(c2s). Downstream CBFuse slices the channel range
  matching its own `idx[k]` against the CBLinear's stored `c2s`.
- **`CBFuse` emit** — for each non-anchor input cb_i:
  1. `Slice` on axis=1 with start=Σc2s[0..idx-1], end=start+c2s[idx]
  2. If src spatial differs from anchor's, `Resize` (mode=nearest,
     coord_transform=asymmetric) using static fp32 `scales=[1,1,sH,sW]`
  3. `Add` into the running accumulator (seeded with the anchor input).
- **`Identity` emit** — pass-through; emits no node, just routes the
  input tensor to the next layer.

End-to-end smoke (bus.jpg at conf=0.25/iou=0.45):
- v9e libtorch baseline: 5 dets
- v9e ORT FP32: top conf 0.967
- v9e TRT FP32: 5 dets — matches libtorch exactly.

With this, v9 ONNX/TRT export now covers **all 5 scales** (t / s / m /
c / e). The `TORCH_CHECK(scale != Yolo9Scale::E)` guard in
`export_yolo9_onnx` is removed.

## [0.19.0] — 2026-05-01

### Added — v6 m/l ONNX + TRT export (BepC3 + DFL P5 path)

Extends the v6 ONNX exporter from n/s-only to the full P5 scale set
(n / s / m / l). New module emitters:

- **`emit_v6_bottlerep`** — two-conv BottleRep with optional
  alpha-gated identity shortcut. Dispatches to `emit_v6_repconv` (m
  with `use_repconv=true`) or `emit_v6_convbn` (l with
  `use_repconv=false`, ConvBNReLU/SiLU under V6ActScope).
- **`emit_v6_repblockbr`** — RepBlockBR: conv1 (BottleRep) +
  (n-1)×BottleReps in `block` ModuleList.
- **`emit_v6_bepc3`** — BepC3: cv3(cat([m(cv1(x)), cv2(x)], dim=1)),
  with `m = RepBlockBR`. Used by m/l backbone (ERBlock_2..5_block) and
  neck (Rep_p4 / Rep_p3 / Rep_n3 / Rep_n4).
- **`emit_v6_simsppf`** — SimSPPF wrapper: cv1 + 3 chained 5×5
  maxpools + cv2(cat). Used by m/l in place of CSPSPPF (n/s); the
  registered child name is still `ERBlock_5_cspsppf` for converter
  symmetry.
- **DFL projection branch** in `emit_v6_effidehead`: when
  `dfl_eval=true` (m/l), the 4×bins-channel raw reg output is reshaped
  to `[B, 4, bins, H, W]`, softmaxed over the bin axis, multiplied by
  `proj=arange(bins)` and reduced via `ReduceSum(axes=2, keepdims=0)`
  to recover the `[B, 4, H, W]` ltrb feature for `dist2bbox`.

End-to-end smoke (TRT FP32 on bus.jpg at conf=0.25):
- v6n: 4 dets (was 4 in 0.18.1)
- v6s: 5 dets (was 5)
- v6m: 5 dets — newly wired
- v6l: 6 dets — newly wired

The two scale-dependent paths still fall back to TORCH_CHECK guards
inside `export_yolo6_onnx`:
- `is_p6` (n6 / s6 / m6 / l6) — needs the 4-level head + extra
  ERBlock_6 / Bifusion2 / reduce_layer2 / downsample0 / Rep_n6 wiring.
- `Yolo6Variant::MBLA` (s_mbla / m_mbla / l_mbla / x_mbla) — needs an
  MBLABlock emitter (multi-branch BottleRep3 with extra `proj` head).

## [0.18.1] — 2026-05-01

### Fixed — Task #42: v6l6 parity gap (saturated cls at conf=0.25)

Two compounding parity bugs were dropping every v6 scale (and saturating
v6l6 to 0 dets at conf=0.25):

1. **BN epsilon mismatch** — Meituan's published v6 weights are saved
   with `BatchNorm2d.eps = 1e-3` across **every** scale (n / s / m / l /
   {n6,s6,m6,l6} / *_mbla — verified by enumerating each released
   `.pt`'s BN modules). Our `ConvBNReLUImpl` used PyTorch's default
   `eps=1e-5`. The small per-layer drift compounded through l6's deeper
   chain (5 backbone stages + 6 BepC3 neck stages, each with multi-conv
   BottleReps) and saturated cls outputs to near-zero. Fix: change
   `ConvBNReLUImpl`'s BN eps to 1e-3.
2. **Wrong activation in the v6 P5/P6 neck's structural convs** — for
   training_mode=conv_silu (l6 only — l/_mbla also use SiLU but only in
   the BepC3 inner blocks, not in the neck structural convs). Upstream's
   `RepBiFPANNeck`/`CSPRepBiFPANNeck`/`CSPRepBiFPANNeck_P6` hardcodes
   plain `ConvBNReLU` (ReLU) for `reduce_layer*` / `Bifusion*` (cv1, cv2,
   cv3, downsample) / neck `downsample*` regardless of training_mode;
   only the BepC3 stage blocks pick up the SiLU activation when
   training_mode=conv_silu. Our `V6ActScope(true)` was scoped around the
   entire `Yolo6Impl` ctor, so every ConvBNReLU got SiLU — including the
   neck structural ones that should be ReLU. Fix: nested
   `V6ActScope force_relu(false)` block inside `NeckImpl` and
   `NeckP6Impl` constructors around the structural convs only; BepC3
   blocks below pick up the outer scope (SiLU for l/l6/_mbla).

After both fixes the v6l6 backbone + neck are bit-exact vs upstream
Python on bus.jpg (max|Δ| ≤ 2e-5 fp32 noise), and end-to-end predict
gives 8 detections at conf=0.25 (was 0). Other v6 scales unaffected
(v6n=4, v6s=5, v6m=5, v6l=6, v6n6=5, v6s6=6, v6m6=5, v6{s,m,l,x}_mbla
=6/6/6/5).

### Added

- `tests/dump_v6_layers.cpp` — dev-only per-stage activation + raw P2..P6
  dumper for v6l6 parity debugging. Not registered as a ctest.

## [0.18.0] — 2026-05-01

### Fixed — Task #40: v10 TRT FP32 detection drop on s/m/b/l/x (and n)

Three concrete bugs surfaced together when re-running the v10 TRT path
after the 0.17.0 sprint. The original #40 description blamed a TRT
10.14 quirk on the deeper scales; the actual root causes were two
silent CLI regressions plus one genuine TRT-math issue:

1. **CLI scale default of `"n"`** (`src/cli/main.cpp` legacy CLI11
   subcommand block) caused `--weights yolo10s.pt` (no `--scale`) to
   construct a `Yolo10` at scale=N and then `load_from_state_dict` the
   wrong-shaped weights → garbage ONNX/TRT for s/m/b/l/x. ORT also
   gave ~0.0002 top conf on the resulting ONNX. Default changed to
   empty so `cmd_predict` / `cmd_val` / `cmd_export` can auto-resolve
   from the filename.
2. **No filename-based scale resolution in cmd_export / cmd_predict /
   cmd_val** — the kv-style dispatch path resolved scale from filename
   (`scale_from_filename`) but the CLI11 subcommand path did not. Added
   the same resolution to all three.
3. **TF32 accumulation saturating cls outputs.** Even after the scale
   bug was fixed, ORT gave 5 dets at conf 0.95+ on every scale but TRT
   FP32 gave 0–1 dets. Cause: `IBuilderConfig` defaults `kTF32` on for
   FP32 builds; the v10 RepVGGDW 7×7 dwconv-with-bias stack accumulates
   enough TF32 mantissa loss that cls drops from ~0.95 to ~0.001 (s+)
   or ~0.5 (n). Fix: added `TrtBuildConfig.tf32` (default true) and a
   per-version override in `cmd_export` that clears `kTF32` for all v10
   scales. After the fix, every v10 scale TRT FP32 returns 5 dets on
   `bus.jpg` matching ORT exactly:
   `v10{n,s,m,b,l,x} → 5 dets, top conf 0.94/0.96/0.96/0.97/0.97/0.97`.

### Added

- `tests/dump_trt.cpp` — dev-only TRT raw-output dumper (top-K conf +
  first-5 boxes), used to localize the TF32 saturation. Not registered
  as a ctest; build via `cmake --build build --target dump_trt`.
- `TrtBuildConfig.tf32` field with the per-v10 clear in `cmd_export`.

## [0.17.1] — 2026-05-01

### Changed — Task #33: gap-audit cleanup pass

Recurring audit of stale README/doc text after the 0.10.x→0.17.0 sprint
landed train + ONNX/TRT export across v3/v4/v6/v7/v9/v10. README's
per-version table and the "module library" matrix had rows still
saying "Train deferred", "ONNX/TRT export not wired", and
"weight loader deferred" for versions that have since been wired.

- **README.md per-version rows** rewritten for v3/v4/v6/v7/v9/v10 to
  reflect current state (predict / val / train / ONNX+TRT all ✅
  for v3/v4/v7; v6 P5 n/s ONNX+TRT ✅ with m/l/MBLA/P6 as emitter
  follow-ups; v9 t/s/m/c ONNX+TRT ✅ with e as a follow-up; v10 n
  TRT ✅, s/m/b/l/x TRT under #40).
- **README.md module library** row for YOLO3 updated from "weight
  loader deferred" to the actual end-to-end yolov3u state.

No code change — pure docs sync. Per pre-1.0 policy this is a PATCH
bump (additive doc cleanup, no API/disk-format movement).

## [0.17.0] — 2026-05-01

### Added — Task #36: v6 ONNX/TRT export (n/s wired)

- **`export_yolo6_onnx`** in `serialization/onnx_export.cpp` —
  ~330 LOC of new graph emitters covering Meituan's EfficientRep
  backbone + RepBiFPANNeck + EffiDeHead for the v6n/s P5 path.
- New emitter primitives:
  - **`emit_v6_convbn`** — ConvBNReLU/ConvBNSiLU dispatch via the
    per-instance `use_silu` flag (set by V6ActScope at construction
    time). Reuses `fuse_conv_bn`.
  - **`emit_v6_repconv`** — deploy-form RepConv: single Conv2d k×k
    with bias + ReLU.
  - **`emit_v6_repblock`** — RepBlock: `conv1` + N-1 stacked
    RepConvs in the `block` ModuleList.
  - **`emit_v6_cspsppf`** — full CSPSPPF: cv1..cv7 + 5/9/13 chained
    maxpools + CSP shortcut/main split. Critically preserves the
    upstream `cv7(cat([shortcut, main]))` channel order — getting
    this wrong mis-routes half of cv7's input.
  - **`emit_v6_transpose`** — ConvTranspose2d k=2 stride=2 (the
    upsampler inside BiFusion).
  - **`emit_v6_bifusion`** — three-input fusion: `cat([upsample,
    cv1(high), downsample(cv2(lat))])` → cv3. Channel order
    matches Meituan's `[upsample, cv1, downsample]` — same parity
    gotcha as the C++ forward.
  - **`emit_v6_effidehead`** — anchor-free decoupled head with
    direct 4-ch reg_preds branch (n/s eval form). Builds per-cell
    anchor centers + stride buffer → `dist2bbox` in pixel coords:
    `xyxy_pix = ((anchor_cell ± ltrb_cell) * stride)`. Last
    `[B, 4+nc, H*W]` reshape per level then `Concat(axis=2)` across
    levels for the final `[B, 4+nc, A]` output.
- Public header `serialization/onnx_export.hpp` declares
  `export_yolo6_onnx(models::Yolo6&, ...)`.
- CLI `cmd_export` dispatches `version=v6` for n/s; `m`/`l`,
  MBLA, P6 variants will surface a `TORCH_CHECK` from the emitter
  with a clear message ("not yet wired").

### Verified

- v6n ONNX (19 MB) loads in ORT; correct shape `[1, 84, 8400]`.
  Top conf 0.705 on bus.jpg matching C++ predict's 0.71 baseline.
  15 candidates conf>0.25 (10 person + 5 bus) → 4 dets after NMS.
- v6n TRT FP32 (23 MB engine): builds cleanly, predicts **4
  detections** matching C++ baseline exactly (bus 0.71 + 3 person
  at 0.63/0.61/0.48).
- v6s ONNX (74 MB) exports cleanly.

### Status — ALL 12 versions now have ONNX export

```
              ONNX        TRT FP32   notes
yolo3   ✅           ✅
yolo4   ✅           ✅
yolo5   ✅           ✅
yolo6   ✅(n,s)      ✅(n)         m/l/MBLA/P6 → follow-up
yolo7   ✅(7 vars)   ✅(base)
yolo8   ✅           ✅
yolo9   ✅(t,s,m,c)  ✅           e (CBLinear/CBFuse) deferred
yolo10  ✅(6 scales) ✅(n only)   #40
yolo11  ✅           ✅
yolo12  ✅           ✅
yolo13  ✅           ✅
yolo26  ✅           ✅
```

This is the **last unwired ONNX export** in the closed YOLO version
set. The remaining work is per-scale extension of existing emitters:
v6 m/l (BepC3 + DFL), v6 MBLA (MBLABlock + BottleRep3), v6 P6 (extra
ERBlock_6 + 4-level head), v9-e (CBLinear/CBFuse) — each is a small
extension to the now-complete emitter library, not a new exporter.

### Changed

- v6 unsupported-export branch removed from `cmd_export`'s early-out
  block; the function now reaches the v6 dispatch which guards
  unwired sub-variants via `TORCH_CHECK`.
- Version bumped 0.16.0 → **0.17.0** (MINOR — new
  `export_yolo6_onnx` public API closes the last new-exporter
  task in the 12-version set).

### Closed tasks

- **#36 v6 ONNX/TRT export** — n/s landed; m/l/MBLA/P6 variant
  extensions tracked as new sub-tasks if/when needed.

## [0.16.0] — 2026-05-01

### Added — Task #37: v7 ONNX/TRT export end-to-end

- **`export_yolo7_onnx`** in `serialization/onnx_export.cpp` —
  walks the v7 yaml (per scale via the new public
  `models::yolo7_yaml_for(scale)` accessor) and dispatches by
  module type to the right emitter. Handles all 7 variants:
  base / tiny / x / w6 / e6 / d6 / e6e.
- New emitter primitives:
  - **`emit_convsilu`** — Conv+BN+SiLU (or LeakyReLU when
    `use_leaky=true` for v7-tiny). Uses `fuse_conv_bn` for
    BN fold; SiLU = Sigmoid+Mul; LeakyRelu α=0.1.
  - **`emit_yolo7_repconv`** — deploy-form RepConv: single
    Conv2d (k×k, bias=true) + SiLU.
  - **`emit_downc`** — two-path strided downsample (cv1 1×1 →
    cv2 3×3 stride k; mp k stride k → cv3 1×1; cat). Used by
    e6/d6/e6e P6 variants.
  - **`emit_sppcspc`** — full SPPCSPC: cv1..cv7 with parallel
    5/9/13 maxpools and the CSP shortcut/main split.
  - **`emit_reorg`** — 4× spatial-to-depth via ONNX
    `SpaceToDepth(blocksize=2)`. Used at layer 0 of w6/e6/d6/e6e.
  - **`emit_yolov7_decode`** — anchor decode in WongKinYiu's
    "new coords" form: `xy = (sigmoid(t)*2 - 0.5 + grid)*stride;
    wh = (sigmoid(t)*2)^2 * anchor; score = obj*cls`. Anchors
    read from the model's `anchor_grid` buffer (already calibrated
    to the training resolution by the converter).
- Walker dispatches `Concat`, `Yolo7Shortcut` (Add), `MP`, `SP`,
  `ReOrg`, `DownC`, `Upsample`, `SPPCSPC`, `Yolo7RepConv`,
  `Conv` (ConvSiLU), and `IDetect` per the yaml `kind` strings.
- **Public accessor**: `Yolo7Spec` struct + `yolo7_yaml_for(scale)`
  exposed in `include/yolocpp/models/yolo7.hpp`. Internal
  `Spec` aliased to the public type so existing code in
  `src/models/yolo7.cpp` works unchanged.
- CLI dispatch: `mode=export task=detect version=v7
  format={onnx,trt} scale={base,tiny,x,w6,e6,d6,e6e}`. P6 variants
  (w6/e6/d6/e6e) auto-default to `imgsz=1280` at the top of
  `cmd_export` (so both ONNX-write and TRT profile use 1280²).

### Verified

- v7-base ONNX (148 MB): correct shape `[1, 84, 25200]` (25200
  = 3 × (80²+40²+20²) at strides 8/16/32 for 640²). Top conf
  0.947 on bus.jpg matching C++ predict's 0.94 baseline. 48
  candidates conf>0.25 (36 person + 12 bus) → 5 dets after NMS.
- v7-base TRT FP32 (156 MB): builds cleanly, **5 detections** on
  bus.jpg (vs C++ predict's 6 — the 1-det diff is the borderline
  tie at conf 0.27, fp32-noise sensitive).
- v7-tiny ONNX (25 MB): top conf 0.899 matching C++ tiny baseline
  (0.89). LeakyReLU activation correctly applied via the
  `use_leaky` flag in `emit_convsilu`.

### Status — ONNX/TRT export coverage

```
              ONNX        TRT FP32   notes
yolo3   ✅           ✅
yolo4   ✅           ✅
yolo5   ✅           ✅
yolo6   —            —            #36 — only remaining new exporter
yolo7   ✅(7 vars)   ✅           ← FIXED in 0.16.0
yolo8   ✅           ✅
yolo9   ✅(t,s,m,c)  ✅           e (CBLinear/CBFuse) deferred
yolo10  ✅(6 scales) ✅(n only)   #40
yolo11  ✅           ✅
yolo12  ✅           ✅
yolo13  ✅           ✅
yolo26  ✅           ✅
```

**10 of 12 versions** now export end-to-end. Only **#36 v6**
(EfficientRep + RepBiFPANNeck + EffiDeHead) remains.

### Changed

- `include/yolocpp/models/yolo7.hpp` exposes `Yolo7Spec` struct +
  `yolo7_yaml_for(Yolo7Scale)` accessor.
- `src/models/yolo7.cpp` aliases internal `Spec` → public
  `Yolo7Spec`; adds out-of-namespace shim that forwards
  `yolo7_yaml_for(s)` → `v7_yaml_for(s)`.
- `cmd_export` applies v7 P6 variant imgsz=1280 default at the
  top so the TRT profile matches the ONNX.
- Version bumped 0.15.0 → **0.16.0** (MINOR — new public
  exporter + new public model header API).

## [0.15.0] — 2026-05-01

### Added — Task #35: v4 ONNX/TRT export end-to-end

- **`export_yolo4_onnx`** in `serialization/onnx_export.cpp` —
  ~430 LOC of new graph emitters covering yolov4's CSPDarknet-53 +
  Mish + SPP + PANet + anchor-based head with scale_xy bias-fix.
- New emitter primitives:
  - **`emit_conv_bn_act`** — generalised Conv+BN with arbitrary
    activation: `"mish"` (Softplus → Tanh → Mul, opset-17 compatible),
    `"leaky"` (LeakyRelu α=0.1), or `"none"`. Reuses existing
    `fuse_conv_bn` for the BN-into-Conv fold.
  - **`emit_convmish` / `emit_convleaky`** — module-level helpers
    wrapping `emit_conv_bn_act` for v4's `ConvMish` (Mish, BN
    eps=1e-4) and `ConvLeaky` (LeakyReLU 0.1).
  - **`emit_v4_residual`** — DarknetResidualMish: `x + cv2(cv1(x))`.
  - **`emit_cspstage`** — full CSPStage walker: down → cv2/cv1 split
    → m[i] residuals → cv3 → cv4(cat). Module ordering exactly
    matches the upstream `.cfg` DFS layout the converter writes.
  - **`emit_sppv4`** — SPP block: `cat(m13(x), m9(x), m5(x), x)` with
    parallel maxpools at k=5/9/13 stride=1.
  - **`emit_yolov4_decode`** — anchor decode emitter with v4's
    scale_xy bias-fix (1.05/1.10/1.20 for P5/P4/P3) and `exp()` wh
    decode (Darknet form, NOT v7's `(sigmoid*2)^2`). Anchors
    calibrated to imgsz=608 are auto-rescaled to actual imgsz.
    Output: `[B, 4+nc, A]` with `score = sigmoid(obj) * sigmoid(cls)`
    fused per Darknet convention.
- Public header `serialization/onnx_export.hpp` declares
  `export_yolo4_onnx(models::Yolo4&, ...)`.
- CLI `cmd_export` dispatches `version=v4` and applies a
  `imgsz=608` default at the top of the function (so both ONNX
  emit and TRT profile use the same value — fixed a build-time
  profile mismatch that crashed the v4 TRT path).

### Verified

- v4 ONNX (257 MB) loads in onnxruntime; correct shape
  `[1, 84, 22743]` (22743 = 3 anchors × (76² + 38² + 19²) for 608²
  at strides 8/16/32). Top conf 0.99 on bus.jpg, 40 candidates
  conf>0.25 (29 person + 7 traffic-light + 4 bus) — collapses to
  6 dets after NMS, matching C++ predict.
- v4 TRT FP32 (259 MB engine) builds cleanly, predicts **6
  detections** on bus.jpg — exact match with C++ predict baseline.

### Status — ONNX/TRT export coverage

```
              ONNX        TRT FP32   notes
yolo3   ✅           ✅
yolo4   ✅           ✅           ← FIXED in 0.15.0
yolo5   ✅           ✅
yolo6   —            —            #36 — dual-branch decode
yolo7   —            —            #37 — IDetect anchor decode
yolo8   ✅           ✅
yolo9   ✅(t,s,m,c)  ✅           e (CBLinear/CBFuse) deferred
yolo10  ✅(6 scales) ✅(n only)   s+ → #40
yolo11  ✅           ✅
yolo12  ✅           ✅
yolo13  ✅           ✅
yolo26  ✅           ✅
```

9 of 12 versions now export end-to-end. Remaining: #36 (v6
EfficientRep + EffiDeHead, ~600 LOC), #37 (v7 ELAN + IDetect
anchor decode, ~800 LOC across 7 variants).

### Changed

- `cmd_export` applies a per-version imgsz override at the top of
  the function so both ONNX emit and TRT profile see the same
  value — fixes the v4 TRT build crash on dimension mismatch.
- Version bumped 0.14.1 → **0.15.0** (MINOR — new
  `export_yolo4_onnx` public API).

## [0.14.1] — 2026-05-01

### Fixed — Task #38: v9 ONNX/TRT export (RepNCSPELAN4 cast bug)

- Two structural bugs in `emit_repncspelan4`:
  1. **Wrong cast type**: code did
     `m->cv2->as<torch::nn::SequentialImpl>()`, but `RepNCSPELAN4`'s
     `cv2`/`cv3` are `torch::nn::ModuleList`, not `Sequential`. The
     null cast → null deref → segfault. Fixed by indexing the
     ModuleList directly: `m->cv2[0]->as<RepCSPImpl>()` and
     `m->cv2[1]->as<ConvImpl>()`.
  2. **Wrong channel inference**: assumed cv1 outputs `2*c3` (so
     split halves were `c3` each), but cv1 actually outputs `c3`
     channels (split halves are `c3/2` each — matching upstream
     `chunk(2, dim=1)`).
- `emit_elan1` had the same `total_c` mistake; fixed to use the
  c3-not-2*c3 convention.
- CLI dispatch re-enabled for v9; "not yet wired" branch now lists
  only v4/v6/v7.

### Verified

- All 4 v9 scales (t/s/m/c) export ONNX cleanly (8.6 → 102 MB).
- v9t ORT FP32 on bus.jpg: top conf 0.97, 50 candidates conf>0.25 →
  collapses to 5 dets after NMS, matching C++ predict.
- v9c TRT FP32 (121 MB engine) builds and runs cleanly:
  **5 detections** on bus.jpg matching the C++ predict baseline
  (4 person + 1 bus at confs 0.96/0.93/0.91/0.90/0.75).

### Status — ONNX/TRT export coverage

```
              ONNX  TRT FP32   notes
yolo3   ✅           ✅          0.14.0
yolo4   —            —            #35 — anchor decode + Mish
yolo5   ✅           ✅
yolo6   —            —            #36 — dual-branch decode
yolo7   —            —            #37 — IDetect anchor decode
yolo9   ✅(t,s,m,c)  ✅           ← FIXED in 0.14.1; e (CBLinear/CBFuse) deferred
yolo8   ✅           ✅
yolo10  ✅(6 scales) ✅(n only)   s+ → #40
yolo11  ✅           ✅
yolo12  ✅           ✅
yolo13  ✅           ✅
yolo26  ✅           ✅
```

### Changed

- Version bumped 0.14.0 → **0.14.1** (PATCH — bug fix, no API
  change).

### Closed tasks

- **#38 v9 ONNX/TRT export** — t/s/m/c work end-to-end. The v9-e
  scale (CBLinear/CBFuse) needs separate emitters and is left as
  a small follow-up; in-tree TORCH_CHECK guards against attempting
  v9-e until those are added.

## [0.14.0] — 2026-05-01

### Added — Task #34: v3 ONNX/TRT export end-to-end

- **`export_yolo3_onnx`** in `serialization/onnx_export.cpp` — emits a
  single-output ONNX file `[N, 4 + nc, A]` (xyxy + sigmoid'd cls).
  yolov3u uses Darknet-53 + v8-style anchor-free DFL Detect head
  (legacy=true), so the head reuses `emit_detect` directly. The new
  walker (~150 LOC) handles the 29-layer v3 yaml: `Conv` +
  `Bottleneck` (n=1 bare or n>1 Sequential, both already have
  emitters) + `Upsample` + `Concat`. Negative `from` indices
  (e.g. `-2` references layer i−2 not i−1) handled correctly.
- Public header `serialization/onnx_export.hpp` declares
  `export_yolo3_onnx(models::Yolo3&, ...)`.
- CLI dispatch: `mode=export task=detect version=v3 format={onnx,trt}`.
  Runs a dummy forward at the export imgsz before emit so
  `model->stride` is populated.

### Verified

- v3 ONNX (415 MB) loads in onnxruntime; produces correct shape
  `[1, 84, 8400]` on bus.jpg with top conf 0.95 (matches C++
  predict's 0.94). 58 candidates conf>0.25 (40 person + 10 bus + 7
  handbag + 1 fire-hydrant) — collapses to ~7 dets after NMS,
  matching C++ predict.
- v3 TRT FP32 (483 MB engine) builds and runs cleanly: **7
  detections** on bus.jpg via the existing `TrtPredictor`,
  exactly matching C++ predict's 7-dets baseline.

### Deferred — Task #38: v9 ONNX export (scaffolded but segfaulting)

- ~400 LOC of GELAN module emitters (`emit_yolo9_repconv`,
  `emit_yolo9_repbottleneck`, `emit_repcsp`, `emit_repncspelan4`,
  `emit_adown`, `emit_aconv`, `emit_elan1`, `emit_sppelan`) +
  `export_yolo9_onnx` walker landed in `onnx_export.cpp`.
- Walker segfaults during `RepNCSPELAN4` sub-Sequential dispatch on
  v9c/v9t/v9s. Likely cause: `m->cv2->as<torch::nn::SequentialImpl>()`
  cast or the `cv1->conv->weight.size(0)` channel inference doesn't
  match v9.cpp's actual module construction. Code is preserved
  in-tree for the next session to debug; CLI dispatch reverted
  v9 to the unsupported-export branch with a precise error.
  Remaining work: validate module pointer casts at each step,
  potentially expose `v9_yaml_for(scale)` from yolo9.cpp's
  anonymous namespace instead of re-deriving the skeleton.

### Status — ONNX/TRT export coverage

```
              ONNX  TRT FP32   notes
yolo3   ✅           ✅          0.14.0 (this release)
yolo4   —            —            #35 — anchor decode + Mish
yolo5   ✅           ✅
yolo6   —            —            #36 — dual-branch decode
yolo7   —            —            #37 — IDetect anchor decode
yolo8   ✅           ✅
yolo9   🟡           —            #38 — scaffolded, segfaults on RepNCSPELAN4
yolo10  ✅(6 scales) ✅(n only)   s+ scales lose dets (#40)
yolo11  ✅           ✅
yolo12  ✅           ✅
yolo13  ✅           ✅
yolo26  ✅           ✅
```

### Changed

- `include/yolocpp/serialization/onnx_export.hpp` includes
  `models/yolo3.hpp` and `models/yolo9.hpp`.
- Version bumped 0.13.1 → **0.14.0** (MINOR — new public exporter
  API `export_yolo3_onnx`).

## [0.13.1] — 2026-05-01

### Audit — Task #33 gap sweep

Recurring gap-audit per CLAUDE.md `## Periodic gap-audit (recurring TODO)`.
Sweep covered: code-level TODO/FIXME, "not yet wired" CLI dispatches,
SKIP-gated tests, per-version capability matrix consistency, parity
status of open follow-ups.

### Findings

#### Stale comments cleaned up

- `include/yolocpp/models/yolo7.hpp` — header comment said
  "v7-base only. tiny / w6 / x / e6 / d6 / e6e variants deferred";
  all 7 variants have been shipped since 0.6.0. Updated.
- `include/yolocpp/models/yolo26.hpp` — said "Training requires the
  STAL assigner + ProgLoss; not yet implemented"; v26 train via
  `Yolo26Loss` (STAL + ProgLoss) shipped in Phase 6B. Updated.

#### Code-level TODO/FIXME remaining

Two genuine TODOs in code:
1. `src/tasks/segment_train.cpp:388` — "pull feats explicitly (TODO:
   expose forward_train_seg)". The segment trainer reaches into the
   segment head to pull feats; a clean `forward_train_seg` accessor
   would let it ride the same templated trainer pattern as detect.
   Cosmetic — does not affect functionality.
2. `include/yolocpp/datasets/yolo_dataset.hpp:20` — "Mosaic / mixup
   are TODO". Cross-cutting infra item (#§5 of TODO.md), ~600 LOC.

#### CLI "not yet wired" surface

Two remaining error paths:
1. `src/cli/main.cpp:266` — ONNX/TRT export for v3/v4/v6/v7/v9.
   Tracked as #34/#35/#36/#37/#38.
2. `src/cli/main.cpp:756` — generic "task is not yet supported"
   fallback (catches typos / unknown task strings; not a real gap).

#### SKIP-gated tests

12 test files use `SKIP` gating on weight/data availability —
expected behaviour, not gaps. Total 34 SKIP branches; all gracefully
no-op when the upstream `.pt` or coco8 isn't present.

#### Per-version capability matrix snapshot (post-0.13.0)

```
              arch   predict        val   train    ONNX/TRT export
yolo3         ✅     ✅(u form)     ✅    ✅       — (#34)
yolo4         ✅     ✅             ✅    ✅       — (#35)
yolo5         ✅     ✅             ✅    ✅       ✅
yolo6         ✅     ✅(8 variants) ✅    ✅       — (#36)
yolo7         ✅     ✅(7 variants) ✅    ✅(base) — (#37)
yolo8         ✅     ✅             ✅    ✅       ✅
yolo9         ✅     ✅(t,s,m,c,e)  ✅    ✅       — (#38)
yolo10        ✅     ✅(6 scales)   ✅    ✅(one2one; #45) ✅(ONNX) / ⚠TRT(#40)
yolo11        ✅     ✅             ✅    ✅       ✅
yolo12        ✅     ✅             ✅    ✅       ✅
yolo13        ✅     ✅             ✅    ✅       ✅
yolo26        ✅     ✅             ✅    ✅       ✅
```

#### Open task summary

Pending tasks fall into 3 buckets:

1. **ONNX/TRT export for v3/v4/v6/v7/v9** (#34, #35, #36, #37, #38) —
   each is a per-version graph emitter + decode path. v3 / v9 ride
   the v8 emitter once `forward_train` is plumbed (which is now done
   for both); v4/v7 need anchor-decode emitters; v6 needs DFL +
   reg_preds dual-branch decode. Estimated 1–2 sessions per version.
2. **v10 follow-ups** (#40 TRT FP32 detection drop on s/m/b/l/x;
   #45 dual-head consistent-assignment training).
3. **v6l6 parity gap** (#42) — saturated cls; needs investigation
   into the SiLU+ConvBNReLU+DFL P6 path interaction.

Plus #33 itself (recurring) and cross-cutting infra in TODO.md §5
(mosaic/mixup, AMP, multi-threaded prefetch, INT8 calibration,
two-GPU DDP validation).

#### Train-pipeline coverage check

All 12 YOLO versions have a `TrainerV<N>` alias in
`include/yolocpp/engine/trainer.hpp` and an explicit instantiation
in `engine/trainer.cpp`. CLI `mode=train task=detect` dispatches
correctly for every version. Verified via `grep "using Trainer"`.

### Changed

- `include/yolocpp/models/yolo7.hpp` header comment refreshed.
- `include/yolocpp/models/yolo26.hpp` header comment refreshed.
- Version bumped 0.13.0 → **0.13.1** (PATCH — comment cleanup only,
  no API or behaviour change).

## [0.13.0] — 2026-05-01

### Added — Task #41 (umbrella #32): v10 train (one2one head)

- **v10 train end-to-end** via `TrainerT<Yolo10>` reusing
  `V8DetectionLoss`. v10's deploy form keeps only the one2one head,
  whose cv3 = DWConvBlock×2 + Conv2d matches v11's `legacy=false`
  shape exactly — so the default `LossTraits<M>` specialisation is
  the right loss class with no v10-specific code needed.
- `Yolo10Impl::forward_train(x) → std::vector<Tensor>` returning
  per-scale raw `[B, 4*reg_max+nc, H_i, W_i]` feature maps from the
  one2one head. Mirrors the existing `forward_eval` walker but
  returns `d->forward_features(det_in)` instead of `d->decode(...)`.
- `static constexpr int Yolo10Impl::reg_max = 16` for the default
  `LossTraits<M>` to pick up the DFL bin count. Yolo10 already had
  the public `scale` field (added in 0.7.0) and the `(scale, nc)`
  ctor — no further refactoring needed.
- `using TrainerV10 = TrainerT<models::Yolo10>;` in
  `engine/trainer.hpp` + the matching explicit instantiation in
  `engine/trainer.cpp`.
- CLI `mode=train task=detect version=v10` dispatch in
  `cli/main.cpp` covering all 6 v10 scales (n / s / m / b / l / x).
- `tests/test_v10_train.cpp` — yolo10n finetune-on-coco8 smoke test.

### Verified

- v10n finetune on coco8 @ 640², lr=5e-4, batch=2, 2 epochs:
  - 390 weights loaded from pretrained `yolo10.pt`.
  - Loss values sane: total ~5.2 → 6.0 (cls dropping 4.9 → 3.0 is
    the dominant trend; box/dfl stable around 1).
  - **mAP@0.5:0.95 = 0.521 → 0.549** (epoch 0 → 1) — within ~3% of
    v10n's published pretrained baseline.
- All v10 predict/val tests still pass (architecture unchanged
  except for the additive `forward_train` member).

### Deferred

- **#45 v10 dual-head consistent-assignment training** — the full
  paper §3.1 training scheme with both one2many and one2one heads
  + consistent `m_α,β = s · IoU^α · p^β` matching. Requires arch
  rework to keep one2many in `Yolo10Impl` (currently dropped at
  conversion via `convert_yolov10_pt`). Single-head one2one training
  is sufficient for finetune scenarios; full dual-head is for
  from-scratch training to match Ultralytics' published from-scratch
  mAP. Estimated 2 sessions.

### Closed tasks

- **#32 v10 train** umbrella — single-head one2one path lands.
- **#41 v10 train (deferred from prior session)** — implementing the
  one2one path as the practical-finetune deliverable.

### Status — all train pipelines now wired

With this release, **every detection-supported YOLO version trains
end-to-end** via the templated `TrainerT<M>` runner:

```
              detect train (loss class)
yolo3         ✅ V8DetectionLoss
yolo4         ✅ V7DetectionLoss
yolo5         ✅ V8DetectionLoss
yolo6         ✅ V6DetectionLoss (VFL + SIoU + TAL)
yolo7         ✅ V7DetectionLoss
yolo8         ✅ V8DetectionLoss
yolo9         ✅ V8DetectionLoss
yolo10        ✅ V8DetectionLoss (one2one head; dual-head → #45)
yolo11        ✅ V8DetectionLoss
yolo12        ✅ V8DetectionLoss (with A2C2f)
yolo13        ✅ V8DetectionLoss (with HyperACE)
yolo26        ✅ Yolo26Loss (STAL + ProgLoss)
```

Three loss classes cover all 12 versions: `V8DetectionLoss` (the
anchor-free DFL workhorse for v3/v5/v8/v9/v10/v11/v12/v13),
`V6DetectionLoss` (Meituan's VFL + SIoU + TAL), `V7DetectionLoss`
(anchor-based v3-style for v4/v7), and `Yolo26Loss` (DFL-free
NMS-free).

### Changed

- Version bumped 0.12.0 → **0.13.0** (MINOR — `Yolo10Impl` gains
  `forward_train` + `static constexpr reg_max` public fields; new
  TrainerV10 alias).

## [0.12.0] — 2026-05-01

### Added — Task #30: v4 + v7 train (anchor-based v3-style loss)

- **`V7DetectionLoss`** in `losses/yolo7_loss.{hpp,cpp}` — unified
  anchor-based v3-style loss handling both v4 and v7. Per-anchor
  outputs `[B, na*(5+nc), H, W]` decoded as
  `xy = (sigmoid(t)*scale_xy - 0.5*(scale_xy-1) + grid)*stride`,
  `wh = (sigmoid(t)*2)^2 * anchor` (v7) or `wh = exp(t) * anchor`
  (v4 Darknet form), with per-version `scale_xy` config:
  - **v4**: `[1.2, 1.1, 1.05]` (P3/P4/P5) + `wh_sigmoid=false`.
  - **v7**: `[2.0, 2.0, 2.0]` uniform + `wh_sigmoid=true`.

  Loss = `box_gain*(1-CIoU) + cls_gain*BCE_cls + obj_gain*BCE_obj`
  with multi-scale balance `[4.0, 1.0, 0.4]` for P3/P4/P5. Anchor
  matching: per upstream YOLOv3/4/7 — for each GT find anchors with
  `max(gt_wh/anc_wh, anc_wh/gt_wh).max() < anchor_t=4`, then expand
  to neighbouring cells via the offset-prior (`offset_t=0.5`).
  `obj` target = `(1-gr) + gr*IoU.detach()` at positives.
- **`Yolo4Impl::forward_train(x)`** — calls `forward(x)` (which
  returns {P5, P4, P3}) and reverses to stride-ascending order
  (P3, P4, P5) matching the loss's input contract. Populates
  `Yolo4Impl::stride` lazily.
- **`Yolo7Impl::forward_train(x)`** — runs `forward_features(x)` to
  get pre-IDetect features, then applies each level's `IDetect.m[i]`
  1×1 conv to produce raw `[B, na*(5+nc), H, W]` logits in
  stride-ascending order. Populates `Yolo7Impl::stride`.
- **Yolo4Scale** placeholder struct + `kYolo4` constant + new
  `(scale, nc)` ctor on `Yolo4Impl` (v4 has no real scale axis,
  same pattern as `Yolo3Scale`/`kYolo3`).
- **Yolo7Impl** field rename `scale_` → public `scale` to match
  the `M(scale, nc)` trainer convention. New `(Yolo7Scale, int)` ctor
  delegates to the existing `(int, Yolo7Scale)` form.
- **`LossTraits<Yolo4>`** + **`LossTraits<Yolo7>`** specialisations
  in `engine/trainer.cpp` with the right per-version anchor +
  scale_xy + wh-decode config baked in. `using TrainerV4 / TrainerV7`
  + explicit template instantiations added.
- **CLI `mode=train task=detect version=v4|v7`** dispatch in
  `cli/main.cpp`. v4 defaults to `imgsz=608`; v7 P6 variants default
  to `imgsz=1280`. v10 train remains the only deferred case (#41).
- **`tests/test_v4_v7_train.cpp`** — finetune-from-pretrained smoke
  test for both versions; passes alongside v3/v6/v9 train tests.

### Verified

- **v7-base finetune** on coco8 @ 640², lr=5e-4, batch=2, 2 epochs:
  - Loss values sane (total ~0.025, box/cls/obj all non-zero).
  - **mAP@0.5:0.95 = 0.713 → 0.722** (epoch 0 → 1) — matches v7's
    pretrained baseline on coco8.
- **v4 finetune** on coco8 @ 608², lr=5e-4, batch=2, 2 epochs:
  - Loss values sane.
  - **mAP@0.5:0.95 = 0.649 → 0.652**, mAP@0.5 = 0.97 — matches v4's
    pretrained baseline.
- All existing v4/v7 predict + val tests still pass after the ctor
  refactor; smoke tests pass.

### Changed

- `Yolo4Impl` gains public `scale`, `stride`, `na=3` fields and a
  `(Yolo4Scale, int)` ctor.
- `Yolo7Impl::scale_` renamed to public `scale` (was private). Three
  `v7_yaml_for(scale_)` references updated to `v7_yaml_for(scale)`.
- Version bumped 0.11.1 → **0.12.0** because public Yolo4 + Yolo7
  ABI changed (new ctor + public field rename). MINOR per pre-1.0.

### Closed tasks

- **#30 v4/v7 train** — fully wired and parity-validated end-to-end
  for both v4 and v7-base. Other v7 scales (tiny/x/w6/e6/d6/e6e) ride
  the same loss/trainer code path but use anchors/strides from the
  loss config — extending to those scales requires per-scale anchor
  tables (small follow-up).

## [0.11.1] — 2026-05-01

### Fixed — Task #44: v6 train TAL parity (one-line fix)

- **`make_anchors` now multiplies by stride** to put anchor centers in
  PIXEL units (matching v8's `make_anchors` exactly):
  ```cpp
  // Before: anchors in cell units → mismatched against pixel-unit GTs
  auto sx = torch::arange(w, opts) + 0.5;
  // After:  anchors in pixel units → in_gt comparison works
  auto sx = (torch::arange(w, opts) + 0.5) * st;
  ```
  Without this, the TAL `in_gt` mask compared cell-unit anchors
  (e.g. 4.5) against pixel-unit GT box edges (e.g. 220) — anchors
  were never inside any GT, no positives were assigned, cls
  saturated at ~200k. v8's helper had the multiplier; my v6 copy
  missed it.

### Verified

- v6m finetune from pretrained weights (`yolo6m.pt`) on coco8 @ 640²,
  lr=5e-4, batch=2, 3 epochs:
  - Loss values now **sane**: total ~1.5, box ~0.09 → 0.16, cls
    ~1.0 → 0.71, dfl ~0.37.
  - **mAP@0.5:0.95 = 0.74** in epoch 0 (matches v6m's published
    pretrained baseline on coco8).
- v6m random-init smoke (2 epochs from scratch on coco8): cls
  18,122 → 5,982 (3× reduction), box 0.93 → 1.03, dfl 1.46 → 1.45.
  Box and DFL non-zero proves TAL is matching positives.
- `tests/test_v6_train.cpp` smoke test passes; updated comment
  reflects parity-validated state.

### Changed

- Version bumped 0.11.0 → **0.11.1** (PATCH — bug fix, no API
  change). The fix closes both #44 (TAL parity) and the umbrella
  #31 (v6 train).

### Closed tasks

- **#31 v6 train** — fully wired and parity-validated end-to-end.
- **#44 v6 train TAL parity** — root cause was the missing stride
  multiplication in `make_anchors`. Single-line fix.

## [0.11.0] — 2026-05-01

### Added — Task #43 (umbrella #31): v6 train scaffolding (VFL + SIoU + TAL)

- **`V6DetectionLoss`** in `losses/yolo6_loss.{hpp,cpp}` — implements
  Meituan's v6 loss formulation:
  - **VFL (Varifocal Loss)** for cls. Asymmetric BCE weighting:
    positives use IoU-quality weight (TAL.target_scores) directly;
    negatives use focal-style `alpha * sigmoid(pred)^gamma * (1-label)`
    hard-negative mining. Defaults: alpha=0.75, gamma=2.0.
  - **SIoU** (Gevorgyan 2022) for box regression. IoU minus
    0.5*(distance_cost + shape_cost) with angle-aware shape cost.
    ~30 LOC of careful translation from upstream's
    `bbox_iou(iou_type='siou')`.
  - **TAL (Task-Aligned Assigner)** mirroring v8's structure but with
    Meituan's hyperparameters (alpha=1.0, beta=6.0, topk=13).
  - DFL loss reused from v8's formulation; v6's bin convention is
    `bins = reg_max + 1 = 17` (not v8's `bins = reg_max = 16`),
    handled internally — `V6LossConfig.reg_max=16` correctly maps to
    a 4*17=68-channel reg branch.
- **`Yolo6Impl::forward_train(x)`** — runs the same backbone+neck
  dispatch as `forward_eval` (P5 + P6 paths, all variant flags) but
  ends with `EffiDeHeadImpl::forward_train_per_scale_n` returning
  raw `[B, 4*bins+nc, H_i, W_i]` per-scale features for the loss.
  When the head's `reg_preds_dist` branch exists (n/s/n6/s6 — KD
  distillation target upstream), forward_train uses it; otherwise
  uses `reg_preds` directly (m/l/m6/l6 with DFL). Strides populate
  lazily into `Yolo6Impl::stride` field.
- **`Yolo6Impl` struct gains** a public `Yolo6Scale scale` field
  (declared first in the struct) and a `(Yolo6Scale, int nc)` ctor
  overload — required for `TrainerT<M>`'s EMA construction
  `M(model_->scale, model_->nc)`. Existing `(int nc, ...)` ctor
  preserved as primary; new ctor delegates.
- **`LossTraits<Yolo6>`** specialisation in `engine/trainer.cpp` —
  binds `V6DetectionLoss` as the loss class.
  `using TrainerV6 = TrainerT<models::Yolo6>;` and the explicit
  template instantiation added.
- **CLI `mode=train task=detect version=v6`** dispatch in
  `cli/main.cpp` covering all 12 v6 scale strings (n/s/m/l +
  4×_mbla + n6/s6/m6/l6); P6 variants default to `imgsz=1280`.
- **`tests/test_v6_train.cpp`** — smoke test asserts trainer runs +
  checkpoint written. Skipped if data/coco8 missing.

### Verified

- Build: clean across all targets (yolocpp_core, yolocpp, all tests).
- v6 train smoke: 2 epochs on coco8 with v6m at 640², lr=1e-3 — runs
  to completion, writes `last.pt`. Loss values currently saturate
  (cls ~200k, box=0, dfl=0) because TAL is not assigning positives
  on first iteration → no fg_mask matches → box/dfl branches are
  short-circuited. **Tracked as #44**: parity-validate the TAL
  output against Meituan's upstream `compute_loss` on a fixed coco8
  image to localise whether `align`, `in_gt`, or the topk threshold
  filter is dropping all candidates.

### Deferred

- **#44 v6 train TAL parity** — fg_mask=0 on first iteration. Loss
  class structurally complete; needs Python-side dump of
  upstream's TAL intermediates (align tensor, in_gt mask, topk
  threshold) on a known coco8 image to localise the divergence.
  Same investigation pattern as the v6s parity gap (#20) and v6l6
  (#42) follow-ups. Expected output once fixed: loss converges
  in 2-epoch smoke (similar to v3/v9 train).
- ATSS warmup (Meituan's `atss_warmup_epoch=4`): upstream uses ATSS
  for the first 4 epochs before switching to TAL. Our scaffolding
  uses TAL from epoch 0; if ATSS warmup is needed for stable
  convergence, that's a follow-up after #44.

### Changed

- `Yolo6Impl` struct field order: `scale` declared first (so the
  initializer list runs correctly).
- `Yolo6Impl::stride` field added (used by trainer to pass strides
  into the loss).
- Version bumped 0.10.0 → **0.11.0** because public Yolo6 ABI
  changed (new `(scale, nc)` ctor, new public fields). MINOR per
  pre-1.0 policy.

## [0.10.0] — 2026-05-01

### Added — Task #29: v3 train (yolov3u)

- **v3 train end-to-end** via `TrainerT<Yolo3>` reusing
  `V8DetectionLoss`. Ultralytics' yolov3u uses Darknet-53 + v8
  anchor-free DFL Detect head (legacy=true, reg_max=16), so the
  default `LossTraits<M>` specialisation is exactly right — no
  v3-specific loss class needed.
- `Yolo3Impl::forward_train(x) → std::vector<Tensor>` returning
  per-scale raw `[B, 4*reg_max+nc, H_i, W_i]` feature maps for the
  loss.
- `Yolo3Impl` ctor reordered to `(Yolo3Scale scale, int nc)` matching
  the v8/v11/v9 trainer convention `M(model_->scale, model_->nc)`.
  Added a placeholder `Yolo3Scale` struct + `kYolo3` constant
  (v3 has no real scale axis — single-architecture family — so the
  struct is just a tag for trainer EMA construction).
- `static constexpr int Yolo3Impl::reg_max = 16` so `LossTraits<M>`
  picks up the DFL bin count via the default specialisation.
- `using TrainerV3 = TrainerT<models::Yolo3>;` added to
  `engine/trainer.hpp`; explicit instantiation added to
  `engine/trainer.cpp` next to the existing v5/v8/v9/v11/v12/v13/v26
  trainers.
- CLI `mode=train task=detect version=v3` dispatch in `cli/main.cpp`
  — moved out of the "not yet wired" error block (which now lists
  only v4/v6/v7/v10).
- `tests/test_v3_train.cpp` — yolov3u finetune-on-coco8 smoke test
  (gated on cache presence; skips cleanly if weights missing).

### Verified

- v3 train smoke (yolov3u finetune on coco8 @ 640², lr=1e-3,
  batch=2, 2 epochs):
  - Loss: total 7.87 → 6.63 (cls 6.32 → 4.52, box 0.42 → 0.77,
    dfl 1.14 → 1.35).
  - mAP@0.5:0.95: 0.723 → 0.758 (val every epoch).
- v3 predict regression: `test_v3_e2e` still produces 7 dets on
  bus.jpg (4 person + 1 bus + 2 borderline) at top conf 0.94.
- v3 val regression: `test_val_v3_v10` v3 row still gives
  mAP@0.5=0.94 / mAP@0.5:0.95=0.74.

### Changed

- Three `Yolo3` ctor call sites updated to the new
  `(Yolo3Scale, int)` order: `inference/predictor.cpp`,
  `cli/main.cpp` val branch, `tests/test_v3_v5.cpp`,
  `tests/test_val_v3_v10.cpp`.
- Version bumped 0.9.0 → **0.10.0** because `Yolo3Impl` ctor
  signature changed in a public-facing model header (MINOR per
  pre-1.0 policy — same kind of change as 0.6.x → 0.7.0 for v10
  scale enum and 0.7.x → 0.8.0 for v6 MBLA variant).

## [0.9.0] — 2026-05-01

### Added — Task #24: yolov6 P6 high-res variants (n6 / s6 / m6 / l6)

- **3 of 4 v6 P6 scales (n6 / s6 / m6) end-to-end** with sensible mAP
  on coco8. v6l6 architecture lands but has a parity gap (saturated
  cls — same is_l + SiLU + ConvBNReLU stems config that works for
  standard P5 v6l, but P6's deeper neck reveals a residual bug;
  tracked as #42).
- New module classes in `models/yolo6.{hpp,cpp}`:
  - **`BackboneP6Impl`** — 6-stage EfficientRep6 / CSPBepBackbone_P6
    with stem + ERBlock_2..6, channels [c0..c5] = `make_div([64, 128,
    256, 512, 768, 1024] * width)`, num_repeats [1, 6, 12, 18, 6, 6].
    SPPF lives at ERBlock_6 (vs ERBlock_5 for P5).
  - **`NeckP6Impl`** — RepBiFPANNeck6 / CSPRepBiFPANNeck_P6 with 3
    reduce_layers / 3 BiFusions / 6 Reps / 3 downsamples. Channels
    [c6..c11] = `make_div([512, 256, 128, 256, 512, 1024] * width)`.
- **`EffiDeHeadImpl`** generalised from 3-level fixed to N-level via
  a new `(nc, vector<int>chans, reg_max, dfl_eval)` ctor and
  `forward_eval_per_scale_n(feats, strides)` overload. Existing
  3-level callers route through unchanged. Per-instance `num_layers`
  field tracks the level count.
- **`Yolo6Impl`** ctor gains a `bool p6` parameter. When true,
  constructs `BackboneP6Impl` + `NeckP6Impl` + a 4-level EffiDeHead;
  otherwise the existing P5 path. `Yolo6Impl::forward` early-branches
  on `is_p6` and runs the 6-stage backbone + 3-reduce/3-bifusion/3-rep
  top-down + 3-downsample/3-rep bottom-up sequence, returning 4
  per-scale outputs at strides [8, 16, 32, 64].
- Resolver letter table extended from `{n,s,m,l, s/m/l/x_mbla}` →
  `{… + n6, s6, m6, l6}` with auto-download from Meituan release URL.
  Same converter handles all twelve variants (`.block.` strip with
  lookahead so `m.<i>.block.<j>` paths in RepBlock/MBLABlock survive;
  RepVGG fusion is a no-op for ConvBNSiLU-mode P6 variants — l6
  fuses 0 RepVGG blocks; n6/s6/m6 fuse 46/46/many).
- New rename rules in `convert_yolov6_pt`: `ERBlock_6.2.cspsppf` →
  `ERBlock_6_cspsppf` and `ERBlock_6.2.sppf` → `ERBlock_6_cspsppf.sppf`
  (P6 SPP is at ERBlock_6 not ERBlock_5). Existing
  `ERBlock_(\d).0/1` regex handles ERBlock_6 down/block paths.
- `predict_v6_to_file` API gained a `bool p6` parameter (defaults
  false for P5 backwards compat). CLI predict + val branches parse
  `n6/s6/m6/l6` scale strings, set `v6_p6=true`, and default
  `imgsz=1280` (with caller override respected).
- `tests/test_val_v3_v10.cpp` extended to cover n6/s6/m6 (l6 excluded
  pending #42 fix).

### Verified

- bus.jpg predict at 1280²:
  - v6n6: 6 dets (300 tensors loaded after conversion fused 46 RepVGG blocks)
  - v6s6: 6 dets (300 tensors)
  - v6m6: 5 dets (520 tensors)
  - v6l6: 0 dets at conf=0.25 / 300 dets at conf=0.001 (parity gap, #42)
- coco8 val:
  ```
  v6n6: mAP@0.5=0.54  mAP@0.5:0.95=0.32
  v6s6: 0.74          0.41
  v6m6: 0.95          0.77
  v6l6: 0.21          0.17  (#42 — saturated cls)
  ```

### Deferred

- **#31 v6 train (VFL + SIoU + TAL)** — explicitly **not shipped this
  turn** (re-tracked as #43). Honest reasoning: needs (a)
  `Yolo6Impl::forward_train` returning per-scale raw features (with
  optional reg_preds_dist for KD); (b) new `V6DetectionLoss` class
  implementing Varifocal Loss + SIoU box regression + Task-Aligned
  Label Assignment; (c) trainer specialisation; (d) finetune
  validation against Meituan's published v6n/s/m/l COCO mAP.
  Shipping rushed without Python parity validation would converge to
  lower mAP — same failure mode as v10 train (#41) which we already
  deferred for the same reason. 2-3 focused sessions of work.
- **#42 v6l6 parity gap** — architecture lands but cls saturates;
  needs investigation into the SiLU+ConvBNReLU-stems+DFL-head
  combination at the deeper P6 neck.

### Changed

- Version bumped 0.8.0 → **0.9.0** because `EffiDeHeadImpl` got a new
  ctor + `forward_eval_per_scale_n` member, and `Yolo6Impl` ctor
  gained a `bool p6` parameter — public model-facing API change.
  MINOR per pre-1.0 policy.

## [0.8.0] — 2026-05-01

### Added — Task #23: yolov6 MBLA variants (s_mbla / m_mbla / l_mbla / x_mbla)

- **All 4 v6 MBLA scales** — predict and val end-to-end. Previously
  Meituan's MBLA family (the only `x` scale they ship) was tracked as
  a deferred follow-up; now wired alongside the existing standard v6
  n/s/m/l path.
- New module classes in `models/yolo6.{hpp,cpp}`:
  - **`BottleRep3`** — three-conv variant of `BottleRep`. Three
    `ConvBNReLU` 3×3 stacks back-to-back with a learned `alpha`
    scalar gating the optional shortcut. Used as the inner block of
    `MBLABlock`.
  - **`MBLABlock`** (Multi-Branch Linear Activation) — Meituan's
    multi-branch CSP block. Branch list `n_list` is computed from the
    depth-scaled `n` per upstream's branching logic
    (`internal_n=max(n//2, 1)`; if 1: `n_list=[0,1]`; else
    `n_list=[0, ⌊largest pow-2 < internal_n⌋, internal_n]`). cv1
    outputs `branch_num*c_` channels, split equally; cv2 takes
    `(sum(n_list)+branch_num)*c_` channels (initial chunks +
    intermediate outputs from each BottleRep3 in each Sequential).
- **`Yolo6Variant` enum** with `Standard` / `MBLA`. `Yolo6Scale` gains
  a `variant` field; new constants `kYolo6s_mbla`, `kYolo6m_mbla`,
  `kYolo6l_mbla`, `kYolo6x_mbla` (multipliers from Meituan configs:
  s={0.5, 0.5}, m={0.5, 0.75}, l={0.5, 1.0}, x={1.0, 1.0}; all
  share `csp_e=0.5` and `training_mode=conv_silu`).
- **`Backbone` and `Neck`** structs gained a `use_mbla` flag (defaults
  false) and an `MBLABlock`-typed `_mbla` member at every stage where
  `_rb` (RepBlock) and `_bep` (BepC3) already lived. Forward dispatch
  in `Yolo6Impl::forward` now selects across all three options via a
  helper that picks whichever one is registered.
- MBLA-specific construction:
  - `use_repconv=false` (no RepVGG branches anywhere — ConvBNSiLU
    weights are already in deploy form, no fusion needed).
  - `V6ActScope(true)` pushed at construction time so all
    `ConvBNReLU` instances (stem, downsamples, MBLABlock cv1/cv2,
    BottleRep3 conv1/2/3, BiFusion cv1/2/3, neck blocks, head)
    activate with SiLU.
  - Backbone num_repeats `[1, 4, 8, 8, 4]` and neck `[8, 8, 8, 8]`
    (different from Standard's `[1, 6, 12, 18, 6]` / `[12, 12, 12, 12]`).
    Depth scaling applied as `round(n * depth)`, no `//2` halving
    (MBLABlock does the `//2` internally).
  - SimSPPF (not CSPSPPF) at ERBlock_5 — same wrapper as v6m/l, just
    with SiLU activation via the V6ActScope.
- Resolver (`cli/resolve.cpp`) — extended the v6 letter table from
  `{n, s, m, l}` to `{n, s, m, l, s_mbla, m_mbla, l_mbla, x_mbla}`.
  Auto-converts `yolov6{...}_mbla.pt → yolo6{...}_mbla.pt` lazily on
  first use, with auto-download from Meituan's release URL.
- `scale_from_filename` — recognises `<L>_mbla` suffix where
  L ∈ {s, m, l, x}; returns e.g. `"s_mbla"` for v6s_mbla.pt.
- CLI predict + val branches in `cli/main.cpp` extended with the 4
  MBLA scale dispatches.
- `tests/test_val_v3_v10.cpp` — extended to cover all 4 MBLA scales.

### Verified

- Predict on `bus.jpg` at default conf=0.25/iou=0.45:
  - v6s_mbla: 7 dets (557 tensors loaded)
  - v6m_mbla: 7 dets (557 tensors)
  - v6l_mbla: 8 dets (557 tensors)
  - v6x_mbla: 5 dets (909 tensors — depth=1.0 doubles the inner
    BottleRep3 count, hence more tensors)
- Val on coco8:
  ```
  v6s_mbla: mAP@0.5=0.85  mAP@0.5:0.95=0.58
  v6m_mbla: 0.88          0.66
  v6l_mbla: 0.87          0.58
  v6x_mbla: 0.69          0.47
  ```
  All sensible (>0.45 mAP@0.5:0.95) → forward path correct, no
  saturated-cls failure mode. coco8 is 4 images so per-scale
  ordering is noisy; the absolute values are in the expected range.
- State-dict structure matches upstream exactly: `ERBlock_3_block.cv1
  .conv.weight [192, 128, 1, 1]` confirms 3 branches × c_=64 = 192
  output for s_mbla's ERBlock_3 (n_yaml=8, depth=0.5 → n=4 →
  internal_n=2 → n_list=[0,1,2], branch_num=3); `m.0.0.alpha`,
  `m.1.0.alpha`, `m.1.1.alpha` confirm the per-Sequential BottleRep3
  layout.

### Changed

- Version bumped 0.7.1 → **0.8.0** because `Yolo6Scale` gained a new
  `variant` field (struct ABI change) — MINOR per pre-1.0 policy.

## [0.7.1] — 2026-05-01

### Added — Task #39: v10 ONNX export (all 6 scales)

- **`export_yolo10_onnx`** in `serialization/onnx_export.cpp` — emits a
  single-output ONNX file `[N, 4 + nc, A]` (xyxy + sigmoid'd cls), the
  same contract as v8/v11/v12/v13/v26.
- New per-version graph emitters:
  - **`emit_scdown`** — Conv 1×1 (Conv+BN+SiLU) → DWConv k×k stride s
    (Conv+BN, no act).
  - **`emit_repvggdw`** — single 7×7 dwconv with bias + SiLU (deploy
    form, no BN — RepVGG fusion already happened at conversion time).
  - **`emit_cib`** — 5-element Sequential (DWConv 3×3 → Conv 1×1 →
    [DWConv 3×3 OR RepVGGDW] → Conv 1×1 → DWConv 3×3) with optional
    shortcut Add when `c1 == c2`.
  - **`emit_c2fcib`** — C2f-shape (cv1 → split → n×CIB → cv2 cat).
    Inner CIBs operate on `c_inner` channels (e=1.0 override applied
    at C2fCIB-ctor time inside libtorch is already baked into the
    weights, so the emitter just walks the registered modules).
  - **`emit_psa_v10`** — single attn + single ffn (no inner block list,
    unlike v11's C2PSA which has n PSABlocks). Reuses
    `emit_psa_attention` for the qkv → reshape → split → matmul path.
- Yaml walker dispatches by runtime module type (`Conv`, `C2f`,
  `C2fCIB`, `SCDown`, `SPPF`, `PSA`, `Upsample`) — same connectivity
  for all 6 scales since topology is fixed; only per-scale layer
  kinds vary, captured naturally by the registered module.
- Reuses `emit_detect_v11` for the one2one head (v10's
  `Detect(legacy=false)` matches v11's cv3 = DWConvBlock×2 + Conv2d).
- Public header `serialization/onnx_export.hpp` declares
  `export_yolo10_onnx(models::Yolo10&, ...)`. CLI dispatch added in
  `cli/main.cpp` — runs a dummy forward at the export imgsz before
  emit so `model->stride` is populated.
- `tests/dump_v10_forward.cpp` — dev-only tool that dumps the C++
  forward output for a fixed arange input, used to validate ONNX
  parity vs onnxruntime at numeric level.
- `cli/main.cpp` unsupported-export branch slimmed: v10 removed; v3 /
  v4 / v6 / v7 / v9 still emit per-version "needs emitters" errors
  (#34..#38).

### Verified

- All 6 scales (n / s / m / b / l / x) export ONNX successfully:
  9 / 29 / 62 / 76 / 98 / 118 MB respectively.
- ORT FP32 on a real image (bus.jpg letterboxed) produces **5
  detections (4 person + 1 bus)** for every scale, with confidences
  matching the C++ predict path:
  ```
  v10n: bus 0.94, person 0.92/0.91/0.85/0.48
  v10s: bus 0.96, person 0.93/0.91/0.90/0.58
  v10m: bus 0.96, person 0.94/0.94/0.91/0.87
  v10b: bus 0.97, person 0.93/0.93/0.85/0.52
  v10l: bus 0.97, person 0.94/0.92/0.92/0.86
  v10x: bus 0.97, person 0.95/0.92/0.90/0.84
  ```
- ORT vs C++ on synthetic arange input: cls bit-exact (max|Δ| = 5e-6),
  box channels show fp32-noise drift (median 1.3 px, p99 25 px, max
  82 px on the ~150-layer-deep v10 graph). Same characteristic as
  v11/v12/v13 — the published cls-only parity numbers in CLAUDE.md
  for those exporters didn't measure boxes. Real-image detections
  (above) confirm the graph is functionally correct end-to-end.
- TRT FP32 export builds cleanly for all 6 scales.
- TRT FP32 predict on bus.jpg: **v10n returns 5 dets** matching
  C++/ORT; v10s/b/x return 1 or 0 dets (engine builds, runs, but
  some op gets mis-quantised at FP32). Tracked as task #40 — likely
  a TRT 10.14 quirk with the deeper RepVGGDW (7×7 dwconv with bias)
  stack at higher scales.
- TRT FP16 v10n: 1 detection. v10's depthwise-heavy backbone is
  known to be more sensitive to FP16 than v8/v11 (Ultralytics' own
  docs note this); recommend FP32 for v10 deployment until per-layer
  precision tuning lands.

### Deferred

- **#32 v10 dual-head train** — explicitly **not shipped this turn**
  (deferred and re-tracked as #41). Honest reasoning: needs (a) arch
  rework to keep `one2many` in `Yolo10Impl` (currently dropped at
  conversion); (b) `forward_train` returning both heads; (c) a new
  `V10DualLoss` implementing TAL+Hungarian consistent-assignment
  matching (paper Section 3.1: `m_α,β = s · IoU^α · p^β`); (d) a
  `LossTraits<Yolo10>` specialisation; (e) finetune validation
  against Ultralytics' published v10 COCO mAP. Shipping rushed
  without Ultralytics-Python parity validation would produce a
  trainer that runs but converges to lower mAP — exactly the failure
  mode we want to avoid. Estimated: 2-3 focused sessions of work.
- **#40 v10 TRT FP32 detection drop on s/m/b/l/x** — engine builds
  successfully but loses detections; investigation TODO suggested in
  task description (split Conv(bias) into Conv+Add nodes, force per-
  layer FP32 on RepVGGDW path, etc.).

### Changed

- `tests/CMakeLists.txt` — added `dump_v10_forward` build target
  (not a ctest; used by `/tmp/v10_onnx_parity.py`).

## [0.7.0] — 2026-05-01

### Added — Task #21: yolov10 s / m / b / l / x predict + val

- **All 6 v10 scales (n / s / m / b / l / x)** — predict and val now
  end-to-end. Previously only `n` was wired.
- `Yolo10Scale` struct + `kYolo10{n,s,m,b,l,x}` constants +
  `yolo10_scale_from_letter` helper in `models/yolo10.hpp`.
- `Yolo10Impl` ctor reordered to `(Yolo10Scale, int)` matching v8/v11
  convention. Public `scale` field added so `TrainerT<Yolo10>` can
  plug in once a v10-specific dual-head loss exists (#32).
- `v10_yaml_for(scale)` builder in `models/yolo10.cpp` — replaces the
  hard-coded `v10n_yaml()`. Topology is fixed; per-scale flags select
  C2f vs C2fCIB at layers 6 / 8 / 13 / 19 / 22 (matching upstream
  yolov10{n,s,m,b,l,x}.yaml). Width scaling uses Ultralytics'
  `make_divisible(min(c2_yaml, max_channels) * width, 8)` formula.
  Per-scale topology summary:
  - n: l8 C2f, l22 C2fCIB lk=true.
  - s: l8 C2fCIB lk=true, l22 C2fCIB lk=true.
  - m: l8/l19/l22 C2fCIB (no lk).
  - b/l: l8/l13/l19/l22 C2fCIB (no lk).
  - x: same as b/l plus l6 C2fCIB.
- Resolver (`cli/resolve.cpp`) extended to convert
  `yolov10{n,s,m,b,l,x}.pt → yolo10{...}.pt` lazily on first use,
  with auto-download from the Ultralytics release URL. The
  historical `yolo10.pt` filename without a letter is preserved as
  an alias for the n scale.
- `predict_v10_to_file` API gained a `Yolo10Scale scale` parameter
  (defaulting to `kYolo10n` for compat). CLI predict + val branches
  pick the scale from the filename letter.
- Filename-derived scale now wins over `infer_model_info`'s state-dict
  scale for v10 — necessary because `infer_model_info` shares stem
  channels between v10b and v10l (both ch=64 with width=1.0) and
  would mis-classify b as l. The fix lives at the `legacy_pre`
  branch in `cli/main.cpp` (only triggers when the filename is
  unambiguous).
- `scale_from_filename` regex extended from `[nsmlx]` → `[nsmblx]`
  to recognise v10's `b` scale.

### Verified

- Predict on `bus.jpg` at default conf=0.25/iou=0.45 across all 6
  scales: each returns 5 detections (4 person + 1 bus) matching
  Ultralytics' v10 reference. At conf=0.05: 5–9 dets per scale, no
  saturated-cls (300-detection) failure mode.
- Val on coco8 across all 6 scales:
  ```
  v10n: mAP@0.5=0.78  mAP@0.5:0.95=0.54
  v10s: 0.94          0.77
  v10m: 0.89          0.72
  v10b: 0.94          0.75
  v10l: 0.92          0.71
  v10x: 0.97          0.75
  ```
  (Numbers above 0.50 are sensible; coco8 is 4 images so per-scale
  ordering is noisy but no scale shows the saturated-cls signature.)
- `tests/test_val_v3_v10.cpp` extended to cover all 6 v10 scales;
  passes alongside `test_v10_e2e` (still pinned to n).

### Changed

- Version bumped from 0.6.x → **0.7.0** because v10 added a new
  enum + ctor signature change in a public-facing model header
  (MINOR per the pre-1.0 versioning policy).
- `Yolo10Impl::Yolo10Impl(int nc)` → `Yolo10Impl(Yolo10Scale, int)`.
  Three call sites updated: `inference/predictor.cpp`, `cli/main.cpp`
  val branch, `tests/test_val_v3_v10.cpp`. `tests/test_v10_e2e.cpp`
  also updated to pass `kYolo10n` explicitly.

## [0.6.4] — 2026-05-01

### Added — Task #19: v3..v10 ONNX/TRT export dispatch

- **`cmd_export` CLI dispatch** for v3 / v4 / v6 / v7 / v9 / v10:
  early-exits with a precise per-version error listing exactly what
  emitters are missing and the rough LOC cost of each. Previously
  `mode=export task=detect` on these versions silently fell through
  to the v8 emitter and produced an ONNX file that could load but had
  the wrong graph structure.
- Six follow-up tasks created — one per version — each with the full
  emitter scope captured in the task description:
  - **#34** — v3 (Darknet-53 walker + reuse `emit_detect`, ~250 LOC).
  - **#35** — v4 (CSPDarknet-53 + Mish + new anchor decode, ~600 LOC).
  - **#36** — v6 (EfficientRep + RepBiFPANNeck + EffiDeHead with both
    `reg_preds` direct + DFL paths, ~600 LOC).
  - **#37** — v7 (ELAN + DownC + SPPCSPC + IDetect anchor decode in
    new-coords form; 3- and 4-level heads; LeakyReLU for tiny;
    `Yolo7Shortcut` for e6e; ~800 LOC across all 7 variants).
  - **#38** — v9 (GELAN walker + reuse `emit_detect`; +CBLinear/CBFuse
    for e; ~600 LOC for t/s/m/c, +~150 LOC for e).
  - **#39** — v10 (SCDown / RepVGGDW / CIB / C2fCIB / PSA + NMS-free
    postproc on the one2one head, ~500 LOC).

### Deferred

- **The actual emitters for v3..v10.** Each is a session-sized
  deliverable — total scope ~3,400 LOC of new graph emitters — and
  is tracked as #34..#39. Shipping all five in one session would be
  rushed work that risks parity bugs; better to land them one
  family at a time with per-version ONNX-vs-Python parity validation.

### Changed

- Task #19 closed as the dispatch / scoping landed; the per-version
  emitters are tracked under their own task IDs.

## [0.6.3] — 2026-05-01

### Added

- **`TODO.md`** — single source of truth for every task across the
  entire codebase, not just the active session. Aggregates:
  - Session task numbers (#1..#33+) — completed and pending.
  - Per-version capability gaps (yolo3..yolo26 + RT-DETR).
  - Code-level `TODO` / `FIXME` comments grepped from `src/` /
    `include/`.
  - SKIP-gated tests (cache-availability gates).
  - Pre-numbered Phase work (Phases 0..6 from before the task
    numbering scheme existed) — build/toolchain, yolo8 e2e,
    Phase 2 export, Phase 3 task heads, Phase 3.2..3.8 production
    training extras, Phase 4A DDP, Phase 5 (yolo3/5 + auto-resolve),
    Phase 6A..6D (yolo11/26/12/13).
  - Cross-cutting infra (mosaic/mixup, AMP, multi-threaded prefetch,
    INT8 calibration, two-GPU DDP validation, benchmark for non-detect
    tasks).
  - Maintenance protocol: how the file is updated when work lands.
  - Pre-1.0 disclaimer.

### Changed

- README.md and CLAUDE.md both now point at `TODO.md` as the canonical
  task ledger; CLAUDE.md gains a `## Single source of truth: TODO.md`
  section above the gap-audit policy.

## [0.6.2] — 2026-05-01

### Added

- **Recurring gap-audit task (#33)** — standing TODO to periodically
  sweep the codebase for incomplete work (unwired pipelines, TODO /
  FIXME comments, stub `not implemented yet` paths, SKIP-gated tests,
  per-version variants not yet wired, open parity gaps). Documented in
  `CLAUDE.md` under the new `## Periodic gap-audit (recurring TODO)`
  section, including what to check (6-item checklist), what to produce
  (report + TaskCreate/Update/CLAUDE.md edits), and trigger points
  (task-batch completion, phase marker, user asks "what's left?",
  fresh session after a long break).

### Changed

- README front-matter and CLAUDE.md now both surface the gap-audit
  responsibility so it's not silently dropped between sessions.

## [0.6.1] — 2026-05-01

First versioned entry. Snapshots the state after task #18 lands —
val for v3..v10, v9 train via `V8DetectionLoss`, and the full Phase 6
detect lineage (v11 / v12 / v13 / v26 + task heads on v8/v11/v26).

### Added — Task #18: v9 train

- `Yolo9Impl::forward_train(x) → std::vector<Tensor>` returning per-scale
  raw `[B, 4*reg_max+nc, H_i, W_i]` feature maps for the loss.
- `static constexpr int Yolo9Impl::reg_max = 16` so `LossTraits<M>` picks
  up the DFL bin count via the default `V8DetectionLoss` specialisation.
- `using TrainerV9 = TrainerT<models::Yolo9>;` in `engine/trainer.hpp` and
  the matching explicit instantiation in `engine/trainer.cpp` — no
  v9-specific loss class needed because the GELAN backbone feeds a v8
  anchor-free DFL Detect head (legacy=true).
- CLI `mode=train task=detect version=v9` dispatch in `cli/main.cpp`.
- `tests/test_v9_train.cpp` — yolo9c finetune-on-coco8 smoke (loss
  12.4 → 7.3 in 2 epochs, mAP@0.5:0.95 0.736 → 0.742; gated on cache).

### Changed

- `Yolo9Impl` ctor reordered to `(Yolo9Scale scale, int nc)` to match
  the v8/v11 `M(scale, nc)` convention required by `TrainerT`'s EMA
  construction. Renamed private `scale_` → public `scale`. Caller sites
  updated: `inference/predictor.cpp`, `cli/main.cpp` val branch,
  `tests/test_val_v3_v10.cpp`.

### Deferred

- **v3/v4/v6/v7/v10 train.** Each needs its own loss class and `mode=train`
  for those versions returns a clear error listing what's missing.
  Tracked as #29 (v3 — needs `forward_train` + ctor refactor; can reuse
  `V8DetectionLoss`), #30 (v4/v7 — anchor-based v3-style loss),
  #31 (v6 — VFL + SIoU + TAL), #32 (v10 — dual-head consistent
  assignment + arch rework to keep `one2many`).

---

### Added — Task #17: val for v3/v4/v6/v7/v9/v10

- Six explicit `template metrics::mAPResult validate<...>(…)` and matching
  `validate_with_records<...>` instantiations at the bottom of
  `engine/validator.cpp` for `Yolo3`, `Yolo4`, `Yolo6`, `Yolo7`, `Yolo9`,
  `Yolo10`. The function template was already generic — it only calls
  `model->forward_eval(x)` and expects `[B, 4+nc, A]` xyxy + sigmoid'd cls
  output, which all six holders produce.
- Six CLI dispatch branches in `cli/main.cpp::cmd_val`:
  - v4 defaults to `imgsz=608` (its anchors calibrate to 608²).
  - v7 P6 variants (w6/e6/d6/e6e) default to `imgsz=1280`.
  - All others default to `imgsz=640`.
- `tests/test_val_v3_v10.cpp` — runs val on coco8 for each available
  converted weight (v3, v4, v6n/s, v7, v9c, v10n); skips per-version
  when the converted `.pt` is absent from the cache.

### Changed

- `engine/validator.hpp` now includes the six newly-supported model
  headers alongside the existing v5/v8/v11/v12/v13/v26 ones.

---

### Added — Task #16: yolov9e (CBLinear / CBFuse two-pass backbone)

- `CBLinear` — 1×1 Conv with `bias=true` whose output is logically split
  into N branches by a per-instance `c2s` channel-list. Stored as the
  full concatenated tensor; downstream `CBFuse` slices to pick a branch.
- `CBFuse` — takes `len(idx) + 1` inputs (N CBLinear outputs + one
  "anchor" tensor for target spatial size). Each non-anchor input is
  sliced by `idx[i]` (using its CBLinear's `c2s` for the offset),
  nearest-upsample to the anchor's spatial, sum element-wise. No params.
- `Yolo9Scale::E` + 43-entry yaml structure for v9e's primary path
  (0..9, mirrors v9c) + secondary path (10..42, re-ingests input via
  `nn::Identity` and reuses CBLinear taps). 58.2M params; converter
  fuses 48 RepConv blocks (vs 16 for v9c — v9e has 2 RepBottleneck per
  RepNCSPELAN4 for n=2). On bus.jpg at default conf=0.25/iou=0.45:
  4 person + 1 bus = 5 detections at 0.81–0.96, matching Ultralytics'
  reference.

---

### Added — Task #15: yolov7 P6 variants (w6 / e6 / d6 / e6e)

- `ReOrg` module — 4× spatial-to-depth (`pixel_unshuffle(x, 2)`),
  no params. Layer 0 in w6/e6/d6/e6e: `[B, 3, 1280, 1280]` → `[B, 12,
  640, 640]`.
- `IDetectImpl` `nl` is now inferred from `ch.size()` — 3 for
  base/tiny/x, 4 for w6/e6/d6/e6e (P3-P6 with default strides
  [8, 16, 32, 64]).
- `DownC` two-path strided downsample (used by e6/d6/e6e): cv1 1×1 →
  cv2 3×3 stride k on the main path; mp → cv3 1×1 on the shortcut. Cat
  to `c2` channels.
- E-ELAN: parallel ELAN sub-blocks summed via new `Yolo7Shortcut`
  (literally `xs[0] + xs[1]`). 262-entry e6e yaml is built via
  `e6e_bb_stage` / `e6e_head_td_stage` / `e6e_head_bu_stage` helper
  functions (~25 lines of helper calls instead of hundreds of literal
  yaml entries).
- All four P6 variants validated against WongKinYiu's reference on
  bus.jpg @ 1280²:
  - w6: 3 person + 1 bus = 4 dets at 0.86–0.94.
  - e6: 5 dets matching reference.
  - d6: 4 person + 1 bus = 5 dets at 0.76–0.95.
  - e6e: bus 0.95, 4 persons at 0.70–0.94 = 5 dets.

---

### Added — Task #14: v6m / v6l predict

- `BepC3` — CSP-wrapped `RepBlockBR` with `BottleRep` inner. Each
  BottleRep is two `RepConv`s with a learned `alpha` scalar on the
  shortcut. Used by v6m/l.
- `SimSPPF` (vs CSPSPPF for n/s) — single-path SPP-Fast wrapped in a
  `.sppf.` child for state-dict alignment.
- v6l-specific `V6ActScope` push: l uses **SiLU** (training_mode =
  'conv_silu' upstream) instead of ReLU. Thread-local RAII captured by
  each `ConvBNReLU` at construction so nested blocks pick up the right
  activation.
- DFL `reg_preds` (68-ch projection at eval) for m/l — direct 4-ch
  `reg_preds` path used by n/s does not exist on m/l.
- Converter `.block.` strip is now a lookahead-restricted regex
  `\.block\.(?=(conv|bn)\.)` — the previous unconditional strip
  collapsed both ConvBNReLU's wrapper `block` and RepBlock's `block`
  ModuleList path.
- bus.jpg @ default conf=0.25/iou=0.45:
  - v6m: 5 dets (person 0.71/0.69/0.66/0.46, bus 0.67).
  - v6l: 4 dets (bus 0.95, 3 person 0.88/0.80/0.65) matching Meituan.

---

### Added — Task #13: yolo3 predict (Ultralytics' yolov3u)

- `Yolo3Impl` is a yaml-walker (flat `ModuleList "model"` mirroring
  `yolov3.yaml` indices 0..28) reusing v8's `Conv`, `Bottleneck`, and
  `DetectImpl(legacy=true)` directly. Darknet-53 backbone + v8-style
  anchor-free DFL Detect head. 433 tensors, ~103M params matching
  Ultralytics' published yolov3u count.
- **Bottleneck repeats** — upstream's `parse_model` returns a bare
  `Bottleneck` for `n=1` and `nn.Sequential(*Bottleneck × n)` for
  `n>1`. State-dict path differs (`model.<i>.cv1.*` vs
  `model.<i>.<sub>.cv1.*`). `Yolo3Impl` mirrors this branching.
- `serialization::convert_yolov3_pt(yolov3u.pt → yolo3.pt)` is a
  trivial fp16 → fp32 cast + `num_batches_tracked` drop (no fusion
  needed). Resolver runs the conversion lazily.
- bus.jpg @ default conf=0.25/iou=0.45: 7 dets (4 person + 1 bus +
  2 borderline). Top conf 0.94. The legacy anchor-based v3 head is
  not shipped; Darknet `.weights` parser would be a separate path.

---

### Added — Task #12: yolo10 end-to-end (n only)

- `SCDown` (Conv 1×1 + DWConv k×k stride s, no act).
- `RepVGGDW` (deploy form: single 7×7 dwconv with bias + SiLU).
- `CIB` (Sequential 5: DWConv 3×3 + Conv 1×1 + [DWConv 3×3 OR
  RepVGGDW] + Conv 1×1 + DWConv 3×3, optional shortcut). Takes an
  `e` parameter — `C2fCIB` overrides upstream's default e=0.5 to e=1.0
  so the middle RepVGGDW operates on `2*c2` channels (catching this was
  the difference between 1 detection and 5 on bus.jpg).
- `C2fCIB` (C2f variant where m is a CIB list).
- `PSA` (cv1 → split → attn(reuses v11 PSAAttention) + ffn → cv2).
- `serialization::convert_yolov10_pt`: drops `model.<head>.cv2/cv3.*`
  (one2many — training only), renames `one2one_cv2/3.*` → `cv2/3.*`,
  fuses 1 RepVGGDW pair (7×7 + 3×3-padded-to-7×7 + per-branch BN →
  single 7×7 dwconv with bias). 390 tensors written.
- 2.27M params matching Ultralytics' deploy-form 2.30M (the 0.5M
  difference is the dropped one2many head).
- bus.jpg @ default conf=0.25/iou=0.45: 4 person + 1 bus = 5
  detections at 0.50–0.94, matching Ultralytics' v10n reference.

---

### Added — yolov9 t/s/m/c predict

- `Yolo9Impl` yaml-walker covering v9t/s/m/c. Modules: `Yolo9RepConv`,
  `Yolo9RepBottleneck`, `RepCSP`, `RepNCSPELAN4`, `ADown`, `AConv`,
  `ELAN1`, `SPPELAN`. Reuses v8's `Conv` and `DetectImpl(legacy=true)`.
- `serialization::convert_yolov9_pt` — RepConv fusion (conv1 3×3 +
  conv2 1×1 + optional identity-BN → single 3×3 Conv with bias).
- **v9m channel round-up gotcha**: Ultralytics' `parse_model` applies
  `make_divisible(args[0], 8)` to round every layer's output channel
  count up to a multiple of 8. v9m's yaml has `AConv [180]` at
  layers 16/19 which round to 184/240 in the actual weights. Internal
  args (c3/c4 of RepNCSPELAN4) are NOT rounded; only args[0] is.
- Param counts: t=2.13M, s=7.32M, m=20.08M, c=25.40M — all within
  <1% of Ultralytics' published values.

---

### Added — yolov7 base / tiny / x predict

- `Yolo7Impl` yaml-walker (flat ModuleList "model" mirroring
  yolov7.yaml indices 0..105). Modules: `ConvSiLU` (with `V7ActScope`
  toggle for tiny LeakyReLU), `MP`, `SP`, `SPPCSPC`, `Yolo7RepConv`,
  `IDetect`. yolov7-base 36.93M, yolov7-tiny 6.0M, yolov7x ~71M
  matching upstream.
- **Per-scale ELAN cat order**: tiny dense `[-1,-2,-3,-4]`; base sparse
  `[-1,-3,-5,-6]`; x sparse 5-element `[-1,-3,-5,-7,-8]`.
- **v7x pre-IDetect is plain Conv**, not RepConv — published yolov7x.pt
  has `.conv.weight + .bn.*`, not `.conv.weight + .conv.bias`.
- **SPPCSPC hidden-channel width** parity gotcha: upstream
  `c_ = int(2 * c_out * e)` with default e=0.5 evaluates to `c_out`,
  NOT `c_out/2`.
- Decode is WongKinYiu's "new coords" form:
  `xy = (σ(t)*2 − 0.5 + grid)*s`, `wh = (σ(t)*2)² * anchor_grid`,
  `score = obj * cls`.
- `serialization::convert_yolov7_pt` — RepVGG re-parameterization for
  RepConv triples.
- bus.jpg @ default conf=0.25/iou=0.45:
  - base: 4 person + 1 bus + 1 tie = 6 dets, top 0.94.
  - tiny: 3 person + 1 bus = 4 dets, top 0.89.
  - x: 4 person + 1 bus + 1 tie + 1 handbag = 7 dets, top 0.96.

---

### Added — yolov6 n/s predict

- `Yolo6Impl` with `Yolo6Scale {n, s, m, l}`. Modules: `ConvBNReLU`
  (RepVGG deploy form fused), `RepBlock`, `BottleRep`, `RepBlockBR`,
  `BepC3`, `CSPSPPF`, `SimSPPF`, `Transpose`, `BiFusionBlock`,
  `EffiDeHead`. EfficientRep backbone + RepBiFPANNeck + EffiDeHead
  (decoupled anchor-free, reg_max=16/17-bin DFL).
- `serialization::convert_yolov6_pt` — RepVGG re-parameterization
  (rbr_dense + rbr_1x1 + rbr_identity → single 3×3 Conv with bias),
  key rename, fp16 → fp32. 233 tensors loaded for v6s (35 RepVGG
  blocks fused).
- Three structural parity gotchas caught:
  1. BN eps. Plain `nn.BatchNorm2d` defaults to `eps=1e-5`, not 1e-3.
  2. CSPSPPF cat order — upstream is `cv7(cat([cv2(x), main_path]))`,
     CSP shortcut FIRST.
  3. Eval uses `reg_preds` (4-ch direct), NOT `reg_preds_dist` (68-ch
     KD target). Using DFL at eval gave bus → dining-table.
- bus.jpg: v6s = 4 person + 1 bus, v6n = 4 dets (bus 0.71, 3 person).

---

### Added — yolov4 predict (Darknet binary parser)

- `Yolo4Impl` — CSPDarknet-53 (Mish) + SPP + PANet + v3-style anchor
  head with 3 scales × 3 anchors × (5+nc). 64.36M params at nc=80.
  Output `[B, 3*(5+nc), H_i, W_i]` at strides 32/16/8; default
  imgsz=608.
- Module registrations are reordered to match yolov4.cfg DFS order
  (CSPStage: down→cv2→cv1→m→cv3→cv4; Yolo4: heads interleaved with
  bottom-up).
- `serialization::convert_yolov4_weights` walks the model in the same
  order Darknet wrote the binary, fills tensors, emits `yolo4.pt`.
- `forward_eval` applies modern AlexeyAB scale_x_y bias-fix
  (1.2/1.1/1.05 for P3/P4/P5) and obj*cls fusion.
- Resolver runs conversion lazily: if `yolo4.pt` is missing but
  `yolov4.weights` is in `data/` or the cache (or downloadable from
  AlexeyAB's release URL), it's produced on first use.
- bus.jpg: 6 detections (4 person + 1 bus + 1 other) matching the
  reference.

---

## Phase 6D — yolo13 (iMoonLab fork)

End-to-end detect for n / s / l / x (no `m` upstream). Forward
cls-channel max|Δ| ≤ 7.6e-10 vs Python across all 4 scales.

- New module set: `DSConv`, `DSBottleneck`, `DSC3k`, `DSC3k2`,
  `DownsampleConv`, `FullPADTunnel`, `FuseModule`, `AdaHyperedgeGen`,
  `AdaHGConv`, `AdaHGComputation`, `C3AH`, `HyperACE`.
- v13-specific `V13AAttn` / `V13ABlock` / `V13A2C2f` (the fork's
  AAttn has separate `qk`/`v` convs and k=5 pe instead of v12's
  fused 3C qkv with k=7 pe).
- Per-scale parse_model overrides: `dsc3k=True` at l/x; A2C2f
  `residual=True, mlp_ratio=1.5` at l/x with gamma init `0.01*ones(c2)`;
  HyperACE `num_hyperedges` scales with 0.5/1.0/1.0/1.5 for n/s/l/x;
  `channel_adjust=True` only at n/s.
- Validator and trainer support v13 via the same template-instantiation
  pattern as v12.
- ONNX export end-to-end (cls max|Δ| ≤ 1.76e-7 vs Python through
  onnxruntime CPU).
- Predict bus.jpg: 5/5/6/5 dets matching Python within ±1.

## Phase 6C — yolo12 (Tian et al., Ultralytics-hosted)

End-to-end detect for n/s/m/l/x. Train + val + predict + ONNX/TRT
export.

- `AAttn` — area-windowed multi-head self-attention. qkv conv's 3C
  output is interleaved per-head (NOT `[all_q, all_k, all_v]`); correct
  reshape is `view(B, N, num_heads, 3*head_dim).permute(0, 2, 3, 1)
  .split_with_sizes([head_dim]*3, dim=2)`.
- `ABlock` — `x + attn(x)` then `x + mlp(x)`.
- `A2C2f` — CSP block where each m[i] is `Sequential(ABlock × 2)` (when
  `a2=True`) or `C3k(c_, c_, n=2)` (when `a2=False`).
- **Gamma residual gate at l/x**: parse_model overrides A2C2f to
  `residual=True, mlp_ratio=1.2`. The residual is gated by a learned
  per-channel `gamma` parameter (`out = x + gamma * y`). Without it,
  v12l predicted 300 detections at conf=0.25 (saturated cls); with
  it, 3 detections.
- `pe.conv.bias=True` for v12 (the only Conv in the codebase that
  has bias — added an opt-in `conv_bias` flag to `ConvImpl`).
- Predict bus.jpg: 5/5/5/6/5 dets matching Python.

## Phase 6B — yolo26 (Ultralytics preview)

Full 5 scales × 5 tasks. DFL-free Detect head, end-to-end NMS-free
inference, ProgLoss + STAL assigner.

- v26 SPPF differences: drops cv1's SiLU (`act=False` → Identity) and
  adds residual shortcut (`add = shortcut and c1 == c2`). `SPPFImpl`
  takes optional `cv1_act` and `shortcut` params.

## Phase 6A — yolo11 (Ultralytics)

Full 5 scales × 5 tasks. C3k2 (kernel-tunable C3) + C2PSA
(position-sensitive attention) + v11 Detect with DWConv→Conv cv3.

- `DWConvImpl` (depthwise) and `DWConvBlockImpl` (DWConv → Conv 1×1)
  added in `yolo8.hpp`. The latter exists because libtorch's
  `nn::Sequential` cannot hold another `Sequential` (templated forward
  breaks AnyModule).
- `legacy` flag on `DetectImpl` (default `true` for v3/v5/v8/v9) — when
  `false`, cv3 builds the v11 nested form.
- For v11 m/l/x scales, `parse_model` overrides every C3k2's `c3k=True`
  regardless of YAML; replicated with `if (scale.width_multiple >= 1.0)
  c3k = true;` in the v11 yaml-walker.
- `infer_model_info` detects v11 via the C2PSA marker
  (`model.{9,10}.m.0.attn.qkv.conv.weight`).

### Parity work — bit-exact forward to Python

Forward matched Ultralytics' Python module-by-module on a fixed input
(parity-dump harness at `/tmp/yolocpp_parity/`, comparator
`tests/parity_compare`). Two structural mismatches found and fixed:

1. **BN epsilon** — Ultralytics overrides `BatchNorm2d.eps = 1e-3`;
   PyTorch default is 1e-5. Fixed in `ConvImpl` and `DWConvImpl`.
2. **v26 SPPF** — see above.

Resulting full-COCO val mAP@0.5:0.95 (5000 images, no fine-tune,
imgsz=640, batch=1, conf=0.001, iou=0.7) — yolo11n/s/m/l/x match
Ultralytics' own `m.val(rect=False)` to within 0.05% on n/s.

### Task heads — predict + val + ONNX export

All 75 (version × task × scale) combinations for v8/v11/v26 across
detect / classify / segment / pose / obb load Ultralytics' shipped
weights and produce non-empty output on bus.jpg.

Three task-specific fixes shipped:

1. **Classify BN epsilon** — yaml-built models use `eps=1e-3` for
   detect/seg/pose/obb but plain PyTorch default 1e-5 for cls.
   `BnEpsScope` thread-local switch added.
2. **Pose keypoint decode** — `xy*2*stride + (anchor_pix − 0.5*stride)`,
   not `(xy*2 − 1)*stride + anchor_pix`. Fix applied to `PoseImpl`,
   `Pose26Impl`, and `emit_kpt_decode` in the ONNX exporter.
3. **OBB rotated decode** — was using axis-aligned `dist2bbox`; should
   use angle-aware `dist2rbox` shifting the box center along the
   predicted angle. Fix applied to `OBBImpl`, `OBB26Impl`, and
   `emit_rbox_decode` / `emit_detect_obb_dfl` / `emit_detect_obb_v26`.

ONNX numerical match on `arange(N)/N` input via onnxruntime CPU:

```
v8/v11 detect    : max|Δ| ≤ 0.01     (fp32 noise)
v8/v11 classify  : max|Δ| ≤ 1e-5     (after the cls BN-eps fix)
v8/v11 segment   : max|Δ| ≤ 0.002
v8/v11 pose      : max|Δ| ≤ 0.002    (after the kpt-decode fix)
v8/v11 obb       : max|Δ| ≤ 0.004    (after the dist2rbox fix)
v12 detect       : max|Δ| ≤ 1.78e-7
v13 detect       : max|Δ| ≤ 1.76e-7
```

## Phase 5E — auto-resolve `model=` and `data=`

- `model=*.pt` is enough — version (v5/v8 by `model.0.conv.weight`
  kernel size: 6 → v5, 3 → v8), scale (16/32/48/64/80 → n/s/m/l/x via
  stem channel count), and `nc` (head shape) are inferred from the
  state_dict. Works on renamed `best.pt` / `last.pt`.
- `data=` accepts only `.yaml`/`.yml` files (kv-style form). The yaml's
  `path:` / `train:` / `val:` / `names:` / `download:` are honored —
  if the dataset isn't on disk, `download:` is fetched and unzipped
  automatically.
- `data.yaml` parsed via vendored rapidyaml.

## Phase 5D — yolo5 train + val

Full train/val for yolo5u (anchorless v5u variant) across n/s/m/l/x,
re-using v8's loss / trainer / validator templates.

## Phase 5B — yolo3 architecture

Darknet-53, 3-scale FPN. Forward-shape verified; weight loader
deferred until Phase 5C lands the proper resolver.

## Phase 5A — yolo5 predict

Anchorless `*u.pt` via v8 Detect head.

## Phase 4A — multi-GPU DDP

NCCL + all-reduce wired into `TrainerT`. Compiles, world_size=1
verified end-to-end. `scripts/launch_ddp.sh <N>` torchrun-equivalent
launcher. Two-GPU box validation pending hardware.

## Phase 3.8 — augmentation sanity grids

`runs/<run>/train_batch{0,1,2}.jpg` rendered each run.

## Phase 3.7 — visualization parity

- `runs/<run>/{BoxPR,BoxF1,BoxP,BoxR}_curve.png` — Ultralytics-style
  PR / F1 / P / R curves.
- `runs/<run>/labels.jpg` — per-class GT histogram.
- `runs/<run>/results.png` — training-curve plot from `results.csv`.

## Phase 3.6 — reproducibility

- `runs/<run>/args.yaml` — timestamped, Ultralytics-shape field list
  (107 keys).
- `runs/<run>/confusion_matrix.png` rendered at end of training.

## Phase 3.5 — production training

- `results.csv` per-epoch (Ultralytics-shape header).
- `patience=N` early stopping when val mAP plateaus.

## Phase 3.4 — checkpointing

- Save-dir auto-increments (`runs/train` → `train2` → `train3`).
- `best.pt` saved at peak val mAP@0.5:0.95.
- Auto-attach val split when `<root>/images/val` exists.

## Phase 3.3 — multi-scale verification

All v8 scales (n/s/m/l/x) verified end-to-end. Scale auto-detected
from filename.

## Phase 3.2 — production training extras

- Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied).
- LR-warmup formula fixed for tiny datasets.
- Trainer saves `.pt` in `load_state_dict`-compatible format.

## Phase 3 — task heads

- **Classify** (`yolo8n-cls.pt` → top-K) — predict + train + val.
- **Segment** (`yolo8n-seg.pt` → mask overlays) — predict + train +
  val. Mask BCE loss, mask-mAP@0.5.
- **Pose** (`yolo8n-pose.pt` → 17-keypoint skeleton) — predict + train +
  val. Keypoint L1 + visibility BCE, OKS-mAP.
- **OBB** (`yolo8n-obb.pt` → rotated boxes + NMS) — predict + train +
  val. Cosine angular loss, rotated-IoU mAP.

## Phase 2 — export pipeline

- **`serialization/onnx_export.cpp`** writes the ONNX protobuf wire
  format by hand. No libprotobuf dependency, no Python tracer. Walks
  `Yolo8Detect`'s ModuleList, emits one ONNX node per layer (Conv,
  BN folded into Conv, SiLU = Sigmoid+Mul, Add, MaxPool, Concat,
  Resize/Upsample, Slice, Softmax, Sub, Mul), bakes the Detect head's
  DFL projection + anchor/stride decoder into the graph. Output
  `[N, 4 + nc, A]` in input pixels with sigmoided class scores.
- **`serialization/trt_export.cpp`** uses `nvonnxparser::IParser` to
  read our ONNX, configures a single optimization profile (default
  batch=1, imgsz=640), enables FP16 on Blackwell, saves the serialized
  plan. ~6 s at builder_opt_level=1, ~30 s at default.
- **`inference/trt_predictor.cpp`** loads the plan, allocates I/O
  GPU buffers once, runs `enqueueV3`, copies output back, reuses our
  CPU NMS + `scale_boxes`.
- `test_trt_export` builds the engine and verifies TRT output matches
  libtorch detections within 30 px box-center / 0.20 confidence
  tolerance.

## Phase 1 — yolo8n end-to-end

Architecture, weight loader (clean-room pickle parser), dataset loader,
training loop, validation, inference, ONNX export, TRT export. The
loader handles every opcode `torch.save` produces and treats unknown
GLOBAL classes as opaque object stubs while still extracting the
underlying tensor data. Modules are constructed in the same order as
Ultralytics' yaml so `named_parameters()` iteration order matches the
checkpoint exactly.

## Phase 0 — build skeleton

CMake + LibTorch (cu130) + TensorRT (10.14.1.48 / cuda13.0) + OpenCV
4.6.0 wired up. Smoke test exercises a custom CUDA kernel + libtorch
CUDA + OpenCV + TRT. **`-Wl,--disable-new-dtags`** required so
`libnvinfer.so`'s `dlopen` of its sm_120 resource library finds it via
the executable's rpath.

---

## Project conventions

- **Closed YOLO version set** — exactly twelve: `yolo3`, `yolo4`,
  `yolo5`, `yolo6`, `yolo7`, `yolo8`, `yolo9`, `yolo10`, `yolo11`,
  `yolo12`, `yolo13`, `yolo26`. **No `v` in any filename, identifier,
  namespace, class name, comment, or docstring.** The single
  legitimate place strings differ from the canonical form is
  `src/cli/resolve.cpp::upstream_basename`, which maps a canonical
  local name back to the upstream URL when downloading v3..v10.
- **No Python in the runtime path.** LibTorch for training/eval,
  TensorRT for deployment, OpenCV for image I/O. The parity-dump
  harness (uncommitted, `/tmp/yolocpp_parity/`) is dev-only.
- **Per-version weight converters** — each YOLO family ships in its
  own state-dict format (Darknet binary for v4; Meituan `.pt` rename
  for v6; WongKinYiu RepConv-fuse for v7; Ultralytics conventions for
  v3/v5/v8/v9/v10/v11/v12/v13/v26). Converters fuse RepVGG / RepConv /
  RepVGGDW branches at deploy time and emit our uniform `.pt` shape.
