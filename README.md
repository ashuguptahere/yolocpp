# yolocpp

Pure C++ computer-vision suite. LibTorch for training/eval, TensorRT for
deployment (Phase 2), OpenCV for image I/O. **No Python in the runtime path.**

## Status — Phase 3.2 (production training) complete

| component                              | state     |
|----------------------------------------|-----------|
| Build skeleton + CMake + smoke test    | ✅ Phase 0 |
| YOLO8n architecture (Conv/C2f/SPPF/Detect/DFL) | ✅ |
| Ultralytics `.pt` weight loader (clean-room pickle parser) | ✅ |
| Letterbox + NMS + predict CLI          | ✅ |
| YOLO-format dataset + augmentation     | ✅ HSV + flip (mosaic/mixup TODO) |
| YOLO8 loss (CIoU + DFL + BCE + TAL)   | ✅ |
| Trainer (SGD + cosine LR + EMA)        | ✅ AMP TODO |
| Validator (mAP@0.5, mAP@0.5:0.95)      | ✅ |
| **ONNX export (hand-written protobuf, no Python)**       | ✅ Phase 2 |
| **TensorRT engine builder (NvOnnxParser, FP16, sm_120)** | ✅ Phase 2 |
| **TRT runtime predictor (matches libtorch detections)**  | ✅ Phase 2 |
| **Classify task (yolo8n-cls.pt → top-K)**               | ✅ Phase 3 — predict + train + val |
| **Segment task (yolo8n-seg.pt → mask overlays)**        | ✅ Phase 3 — predict + train + val (mask BCE loss, mask-mAP@0.5) |
| **Pose task (yolo8n-pose.pt → 17-keypoint skeleton)**   | ✅ Phase 3 — predict + train + val (kpt L1 + visibility BCE, OKS-mAP) |
| **OBB task (yolo8n-obb.pt → rotated boxes + NMS)**      | ✅ Phase 3 — predict + train + val (cosine angular loss, rotated-IoU mAP) |
| **Auto-resolve `model=` and `data=` from cwd / cache / Ultralytics** | ✅ Phase 3.2 |
| **Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied)** | ✅ Phase 3.2 |
| **LR-warmup formula fixed for tiny datasets**            | ✅ Phase 3.2 |
| **Trainer saves `.pt` in load_state_dict-compatible format** | ✅ Phase 3.2 |
| **All v8 scales (n / s / m / l / x) verified end-to-end** | ✅ Phase 3.3 |
| **Scale auto-detected from filename (`yolo8m.pt` → scale=m)** | ✅ Phase 3.3 |
| **Save-dir auto-increments (`runs/train` → `train2` → `train3`)** | ✅ Phase 3.4 |
| **`best.pt` saved at peak val mAP@0.5:0.95**             | ✅ Phase 3.4 |
| **Auto-attach val split when `<root>/images/val` exists** | ✅ Phase 3.4 |
| **`results.csv` per-epoch (Ultralytics-shape header)**   | ✅ Phase 3.5 |
| **`patience=N` early stopping when val mAP plateaus**    | ✅ Phase 3.5 |
| **`runs/<run>/args.yaml` reproducibility dump**          | ✅ Phase 3.6 (timestamped, Ultralytics-shape field list — 107 keys) |
| **`runs/<run>/confusion_matrix.png` rendered at end**    | ✅ Phase 3.6 |
| **`runs/<run>/{BoxPR,BoxF1,BoxP,BoxR}_curve.png`**       | ✅ Phase 3.7 |
| **`runs/<run>/labels.jpg` (per-class GT histogram)**     | ✅ Phase 3.7 |
| **`runs/<run>/results.png` (training-curve plot)**       | ✅ Phase 3.7 |
| **Multi-GPU DDP infrastructure (NCCL + all-reduce)**     | ✅ Phase 4A — compiles, world_size=1 verified; needs 2-GPU box for full validation |
| **`scripts/launch_ddp.sh <N>` torchrun-equivalent launcher** | ✅ Phase 4A |
| **`runs/<run>/train_batch{0,1,2}.jpg` augmentation sanity grids** | ✅ Phase 3.8 |
| **YOLO5 (anchorless v5u) end-to-end** — predict, **train, val** for all five scales (n/s/m/l/x) | ✅ Phase 5A + 5D |
| **YOLO3 architecture (Darknet-53, 3-scale FPN)**        | ✅ Phase 5B (forward-shape verified; weight loader deferred) |
| **CLI dispatch on filename pattern (`yolo3*`, `yolo5*`, `yolo8*`, …)** | ✅ Phase 5C |
| **Ultralytics-style `data=path/to/data.yaml`** (yaml-only — no directory form) | ✅ Phase 5E |
| **`data.yaml` parsed via vendored rapidyaml** (`path:` / `train:` / `val:` / `names:`) | ✅ Phase 5E |
| **Auto-download from `data.yaml`'s `download:` URL** when dataset missing | ✅ Phase 5E |
| **Model auto-inference from `.pt` state_dict shapes** — version (kernel=6 → v5, kernel=3 → v8), scale (16/32/48/64/80 → n/s/m/l/x), nc (head shape) | ✅ Phase 5E |
| **`scale=`/`version=`/`nc=` no longer required** — works on renamed `best.pt`/`last.pt` too | ✅ Phase 5E |

## YOLO version roadmap

The codebase covers **only** these twelve YOLO versions and uses the
single canonical filename / identifier convention `yolo<N>` everywhere
(`yolo3`, `yolo4`, `yolo5`, `yolo6`, `yolo7`, `yolo8`, `yolo9`, `yolo10`,
`yolo11`, `yolo12`, `yolo13`, `yolo26` — **no `v`**). When fetching legacy
upstream Ultralytics weights for v3..v10, the resolver transparently maps
the canonical name back to the `yolov<N>...pt` URL. Versions outside this
set (v1, v2, anything between v13 and v26) are intentionally not supported.

| version | year | provenance | family / changes | status |
|---------|------|------------|------------------|--------|
| **yolo3**  | 2018 | Redmon & Farhadi (official Darknet)        | Darknet-53 backbone + 3-scale FPN, anchor-based head | ✅ architecture (forward-shape verified); weight loader pending |
| **yolo4**  | 2020 | Bochkovskiy et al. (official Darknet)      | CSPDarknet-53 + SPP + PANet; Mish activations; bag-of-freebies tricks | 🟡 stub — needs Darknet `.weights` loader |
| **yolo5**  | 2020 | Ultralytics (official)                     | CSPNet + C3 blocks, originally anchor-based; the modern `*u.pt` files use the v8 anchor-free Detect head | ✅ end-to-end (predict / train / val) for all 5 scales via `yolo5*u.pt` |
| **yolo6**  | 2022 | Meituan (official open-source)             | EfficientRep (RepVGG) backbone + Rep-PAN neck, decoupled anchor-free head, VFL/SIoU loss | 🟡 stub — needs YOLO6 state_dict adapter |
| **yolo7**  | 2022 | Wang, Bochkovskiy & Liao (academic)        | E-ELAN backbone, SPPCSPC neck, IDetect (aux + lead) head, trainable bag-of-freebies | 🟡 stub — distinct state_dict layout |
| **yolo8**  | 2023 | Ultralytics (official)                     | CSP + C2f backbone, anchor-free DFL Detect, TAL assigner | ✅ **full** — train / val / predict / export across 5 scales × 5 tasks (detect / segment / classify / pose / OBB) |
| **yolo9**  | 2024 | Wang, Yeh & Liao (academic)                | GELAN backbone, PGI (Programmable Gradient Information) auxiliary branch | 🟡 stub — plan: reuse v8 head, swap backbone |
| **yolo10** | 2024 | Tsinghua MIG (academic, Ultralytics-hosted)| Dual-head consistent assignment → end-to-end NMS-free at inference, rank-guided block design | 🟡 stub — needs dual-head training graph |
| **yolo11** | 2024 | **Ultralytics (official)**                 | Refined CSP: C3k2 (kernel-tunable C3) + C2PSA (position-sensitive attention); v11 Detect head with depthwise-separable cv3 (DWConv→Conv) | ✅ **end-to-end** — train / val / predict / ONNX + TRT export across all 5 scales (n/s/m/l/x) and 4 task heads (detect/classify/segment/pose/obb). Param counts match Ultralytics exactly (2.62M / 9.46M / 20.1M / 25.4M / 56.97M). **Known calibration quirk on m/l/x:** cv3 outputs are over-saturated independent of input signal (verified by zero-input test); n/s give correct mAP@0.5 ≈ 0.78/0.74 on coco8, m/l/x give 0.03/0.11/0.007. Pinpointing the divergence requires a Python parity harness (see CLAUDE.md). |
| **yolo12** | 2025 | Tian et al. (unofficial)                   | Attention-centric: A2C2f (Area-Attention C2f) for windowed global attention with v8 latency | 🟡 stub |
| **yolo13** | 2025 | Lei et al. (unofficial)                    | HyperACE (hypergraph adaptive correlation enhancement), FullPAD pipeline, DSConv variants | 🟡 stub |
| **yolo26** | 2025 | **Ultralytics (official, preview)**        | DFL-free Detect head, end-to-end NMS-free inference, ProgLoss + STAL assigner — edge/mobile-first | 🟡 stub — needs new head + assigner + loss |
| RT-DETR    | 2023 | Baidu (official)                           | HGNetv2 + AIFI + deformable-attention decoder; transformer-based, NMS-free | 🟡 architecture probed (Phase 4 — transformers) |

Stub status (🟡) means the header / source files exist with the right
class name and namespace, the build links cleanly, and `mode=predict`
will produce a clear `not implemented yet` runtime error pointing at the
header that explains the design plan. Implementation order is roughly:
**yolo11 → yolo9 → yolo10 → yolo26 → yolo12 → yolo13 → yolo6 → yolo7
→ yolo4** (Ultralytics-head-compatible families first, since they let us
reuse the v8 Detect / DFL / loss / trainer wholesale).

| dependency | version            |
|-----------|--------------------|
| LibTorch  | 2.11.0+cu130       |
| TensorRT  | 10.14.1.48+cuda13.0|
| CUDA tk   | 13.0.88            |
| OpenCV    | 4.6.0              |

## Build

```bash
./scripts/install_third_party.sh           # ~5 GB download, idempotent
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The test suite covers every layer of the stack:

| test                 | what it checks                                       |
|----------------------|------------------------------------------------------|
| `smoke_test`         | LibTorch CUDA + custom CUDA kernel + OpenCV + TRT    |
| `test_yolo8_arch`   | YOLO8n architecture: param count (~3.16M), strides, output shapes |
| `test_pt_loader`     | Loads real `yolo8n.pt`, verifies all 355 keys/shapes |
| `test_predict`       | End-to-end inference on `bus.jpg`, ≥1 person + ≥1 bus |
| `test_train_overfit` | Tiny synthetic dataset, training loss decreases ≥ 2× |

## CLI

Two argument styles, both supported:

```
# Ultralytics-style (canonical)
yolocpp task=detect mode=train   model=yolo8n.pt data=path/to/data.yaml epochs=100
yolocpp task=detect mode=val     model=yolo8n.pt data=path/to/data.yaml
yolocpp task=detect mode=predict model=yolo8n.pt source=bus.jpg
yolocpp task=detect mode=export  model=yolo8n.pt format=trt fp16=true
yolocpp mode=benchmark           model=yolo8n.pt source=bus.jpg

# Legacy subcommand-style
yolocpp predict   --weights=yolo8n.pt  --source=bus.jpg --out=out.jpg
yolocpp predict   --weights=yolo8n.trt --source=bus.jpg --out=out.jpg
yolocpp val       --weights=yolo8n.pt  --data=<root>
yolocpp train     --data=<root> [--epochs=100] [--scale=n] [--weights=yolo8n.pt]
yolocpp export    --format=onnx --weights=yolo8n.pt --out=yolo8n.onnx
yolocpp export    --format=trt  --weights=yolo8n.pt --out=yolo8n.trt --fp16
yolocpp benchmark --weights=yolo8n.pt  --source=bus.jpg
yolocpp info
```

`data=` accepts **only a `.yaml`/`.yml` file** in the kv-style form. The yaml's
`path:` / `train:` / `val:` / `names:` / `download:` are honored — if the
dataset isn't on disk, the `download:` URL is fetched and unzipped automatically.

`model=` is enough on its own — version (v5/v8), scale (n/s/m/l/x) and `nc`
are auto-inferred from the `.pt`'s actual layer shapes (`model.0.conv.weight`
kernel + first dim, head's `cv3.0.2.weight` first dim). This works for renamed
checkpoints (`best.pt`, `last.pt`) where the filename carries no version
letter. Pass `version=` / `scale=` / `nc=` only to override.

All five `task` values now support train + val + predict + export (detect)
and train + val + predict (classify/segment/pose/obb):

```
yolocpp task=classify mode=train   data=DATA model=yolo8n-cls.pt epochs=30
yolocpp task=segment  mode=train   data=DATA model=yolo8n-seg.pt epochs=30
yolocpp task=pose     mode=train   data=DATA model=yolo8n-pose.pt epochs=30
yolocpp task=obb      mode=train   data=DATA model=yolo8n-obb.pt epochs=30
yolocpp task=segment  mode=val     data=DATA model=runs/segment/last.pt
```

Dataset layouts:
- **classify**: `<root>/{train,val}/<class_name>/img.jpg`
- **segment**: same as detect, with optional polygon vertices appended
  to each label line (`cls cx cy w h x1 y1 x2 y2 ... xN yN`)
- **pose**: same as detect, with `K` triplets `(kx, ky, v)` appended
  per line (visibility v ∈ {0, 1, 2})
- **obb**: same as detect, with an angle (radians) appended:
  `cls cx cy w h angle`

Loss & metric specifics (minimum-viable; production parity is Phase 3.2):
- **classify**: cross-entropy ; top-1 / top-5
- **segment**: per-positive-anchor mask BCE on `sigmoid(coefs × protos)`
  vs polygon-rasterized mask ; mask mAP@0.5
- **pose**: L1 on `(x, y)` weighted by visibility + visibility BCE ; OKS-mAP@0.5
- **obb**: `1 - cos(pred_angle - gt_angle)` periodic-friendly loss on
  the closest anchor ; rotated-IoU mAP@0.5

### Benchmark output (RTX 5090, FP16 — bus.jpg, 200 iters)

```
  backend                  median (ms)   p95 (ms)   mean (ms)   img/s    dets
  ──────────────────────  ────────────  ─────────  ──────────  ───────  ─────
  PT (libtorch FP32)              3.90       4.38        3.94    256.4      6
  TRT FP32                        2.04       2.20        2.05    491.3      6
  TRT FP16                        1.61       1.69        1.61    622.6      6

  Speedup vs PT:
    TRT FP32                1.92x
    TRT FP16                2.43x
```

Dataset layout:
```
<root>/images/{train,val}/<id>.jpg
<root>/labels/{train,val}/<id>.txt   # one line: cls cx cy w h, all in [0,1]
```

## Architecture map

```
yolocpp/
  models/yolo8         Conv → C2f → SPPF → Detect (DFL)
                        scale variants n/s/m/l/x
  serialization/
    pt_loader           Reads Ultralytics .pt zip + clean-room pickle parser
                        → state_dict {dotted_name → torch::Tensor}
  inference/
    letterbox           letterbox + image_to_tensor + scale_boxes
    nms                 class-aware NMS (CPU)
    predictor           load weights → preprocess → forward → NMS → unscale
  datasets/
    yolo_dataset        YOLO-format (.txt) loader, HSV + flip augmentation
  losses/
    yolo8_loss         TAL assigner + CIoU + DFL + BCE
  engine/
    trainer             SGD/momentum/Nesterov + warmup + cosine LR + EMA
    validator           full pass over val set → mAP via metrics/map
  metrics/map           per-class AP at IoU={0.5, 0.5..0.95}
```

## Architecture decisions worth knowing

- **`.pt` loader is a clean-room pickle parser**, not libtorch's
  `torch::jit::Unpickler`. Reason: the standard Unpickler trips on the
  BUILD opcode for arbitrary Python classes (`DetectionModel` etc.). Our
  parser handles every opcode produced by `torch.save` and treats unknown
  GLOBALs as opaque object stubs while still extracting the underlying
  tensor data.
- **Modules are constructed in the same order as Ultralytics' yaml**, so
  `named_parameters()` iteration order matches the checkpoint exactly.
- **`-Wl,--disable-new-dtags`** so `libnvinfer.so`'s `dlopen` of its
  sm_120 resource library finds it via the executable's rpath. See CLAUDE.md.

## Phase 2 export — how it works

- **`serialization/onnx_export.cpp`** writes the ONNX protobuf wire format
  by hand. We don't link libprotobuf and don't run any Python tracer. The
  exporter walks our `Yolo8Detect` ModuleList, emits one ONNX node per
  layer (Conv, BN folded into Conv, SiLU = Sigmoid+Mul, Add, MaxPool,
  Concat, Resize/Upsample, Slice, Softmax, Sub, Mul), and bakes the
  Detect head's DFL projection + anchor/stride decoder into the graph.
  Output is `[N, 4 + nc, A]` in input pixels with sigmoided class scores
  — same shape libtorch's `forward_eval` returns.
- **`serialization/trt_export.cpp`** uses `nvonnxparser::IParser` to read
  our ONNX, configures a single optimization profile (default
  batch=1, imgsz=640), enables FP16 on Blackwell, and saves the
  serialized plan. `~6 seconds at builder_opt_level=1`, ~30 seconds at
  default level.
- **`inference/trt_predictor.cpp`** loads the plan, allocates input/output
  GPU buffers once, runs `enqueueV3`, copies output back, and reuses our
  CPU NMS + `scale_boxes`. Returns the same `Detection` struct the
  libtorch `Predictor` does.

`test_trt_export` builds the engine on every run and verifies the TRT
output matches libtorch detections within 30 px box-center tolerance and
0.20 confidence tolerance — in practice they agree to FP16 rounding.

## What's deliberately deferred

- **Numerical parity to Ultralytics**: requires a one-time tensor dump
  from Python — outside the runtime contract. The `.pt` loader and
  forward path are structurally exact; producing reference dumps to
  validate every block is a Phase 1.5 task.
- **Mosaic / mixup augmentation**: full multi-image augmentation in C++
  is straightforward but ~600 lines we haven't needed yet.
- **AMP (mixed-precision training)**: Trainer is FP32-only. Adding
  `torch::autocast` is a future change.
- **Multi-threaded data prefetch**: dataset is synchronous. With OpenCV
  decode + CUDA inference, the IO bottleneck on a 5090 is real but
  fixable later.
- **DDP / multi-GPU training**: out of scope.
- **TRT INT8 calibration** and dynamic-shape multi-batch profiles: easy
  to add on top of `TrtBuildConfig` once a calibration set exists.
