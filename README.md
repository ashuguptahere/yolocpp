# yolocpp

Pure C++ computer-vision suite. LibTorch for training/eval, TensorRT for
deployment (Phase 2), OpenCV for image I/O. **No Python in the runtime path.**

## Status — Phase 3.2 (production training) complete

| component                              | state     |
|----------------------------------------|-----------|
| Build skeleton + CMake + smoke test    | ✅ Phase 0 |
| YOLOv8n architecture (Conv/C2f/SPPF/Detect/DFL) | ✅ |
| Ultralytics `.pt` weight loader (clean-room pickle parser) | ✅ |
| Letterbox + NMS + predict CLI          | ✅ |
| YOLO-format dataset + augmentation     | ✅ HSV + flip (mosaic/mixup TODO) |
| YOLOv8 loss (CIoU + DFL + BCE + TAL)   | ✅ |
| Trainer (SGD + cosine LR + EMA)        | ✅ AMP TODO |
| Validator (mAP@0.5, mAP@0.5:0.95)      | ✅ |
| **ONNX export (hand-written protobuf, no Python)**       | ✅ Phase 2 |
| **TensorRT engine builder (NvOnnxParser, FP16, sm_120)** | ✅ Phase 2 |
| **TRT runtime predictor (matches libtorch detections)**  | ✅ Phase 2 |
| **Classify task (yolov8n-cls.pt → top-K)**               | ✅ Phase 3 — predict + train + val |
| **Segment task (yolov8n-seg.pt → mask overlays)**        | ✅ Phase 3 — predict + train + val (mask BCE loss, mask-mAP@0.5) |
| **Pose task (yolov8n-pose.pt → 17-keypoint skeleton)**   | ✅ Phase 3 — predict + train + val (kpt L1 + visibility BCE, OKS-mAP) |
| **OBB task (yolov8n-obb.pt → rotated boxes + NMS)**      | ✅ Phase 3 — predict + train + val (cosine angular loss, rotated-IoU mAP) |
| **Auto-resolve `model=` and `data=` from cwd / cache / Ultralytics** | ✅ Phase 3.2 |
| **Auto finetune-LR (`lr0=0.001` when `model=*.pt` supplied)** | ✅ Phase 3.2 |
| **LR-warmup formula fixed for tiny datasets**            | ✅ Phase 3.2 |
| **Trainer saves `.pt` in load_state_dict-compatible format** | ✅ Phase 3.2 |
| **All v8 scales (n / s / m / l / x) verified end-to-end** | ✅ Phase 3.3 |
| **Scale auto-detected from filename (`yolov8m.pt` → scale=m)** | ✅ Phase 3.3 |
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
| **YOLOv5 (anchorless v5u) end-to-end** — predict, **train, val** for all five scales (n/s/m/l/x) | ✅ Phase 5A + 5D |
| **YOLOv3 architecture (Darknet-53, 3-scale FPN)**        | ✅ Phase 5B (forward-shape verified; weight loader deferred) |
| **CLI dispatch on filename pattern (`yolov3*`, `yolov5*`, `yolov8*`, …)** | ✅ Phase 5C |
| **Ultralytics-style `data=path/to/data.yaml`** (yaml-only — no directory form) | ✅ Phase 5E |
| **`data.yaml` parsed via vendored rapidyaml** (`path:` / `train:` / `val:` / `names:`) | ✅ Phase 5E |
| **Auto-download from `data.yaml`'s `download:` URL** when dataset missing | ✅ Phase 5E |
| **Model auto-inference from `.pt` state_dict shapes** — version (kernel=6 → v5, kernel=3 → v8), scale (16/32/48/64/80 → n/s/m/l/x), nc (head shape) | ✅ Phase 5E |
| **`scale=`/`version=`/`nc=` no longer required** — works on renamed `best.pt`/`last.pt` too | ✅ Phase 5E |

## YOLO version roadmap

| version | family       | status | notes |
|---------|--------------|--------|-------|
| v1, v2  | Joseph Redmon — Darknet  | not planned | obsolete, no Ultralytics weights |
| v3      | Darknet-53 + 3-scale FPN  | ✅ architecture, weights TBD | original anchor-based head |
| v4      | Bochkovskiy — CSPDarknet53 + PANet | next session | weights ship via Darknet, would need a separate loader |
| **v5**  | Ultralytics CSP + C3      | **✅ end-to-end via `yolov5*u.pt`** | the anchorless v5u variant Ultralytics ships shares v8's Detect head |
| v6      | Meituan — RepVGG backbone | next session | different state_dict layout |
| v7      | Wang et al. — ELAN modules| next session | different state_dict layout |
| **v8**  | Ultralytics CSP + C2f + DFL | **✅ full** (train/val/predict/export, 5 scales × 5 tasks) | reference architecture |
| v9      | Wang et al. — GELAN + PGI auxiliary | next session | Ultralytics ships `yolov9c.pt` / `yolov9e.pt` — should reuse v8 head with GELAN backbone swap |
| v10     | Tsinghua — dual-head NMS-free | next session | needs custom dual-head + trained-by-us inference path |
| v11     | Ultralytics — refined C2f | next session | weights at `yolo11n.pt` |
| v12     | Tsinghua — attention-centric | future | architecture published, parity validation pending |
| v13–v26 | (none canonical) | not planned | the canonical YOLO version line currently ends at v12 |
| RT-DETR | Transformer-based | Phase 4 (transformers) | architecture probed: HGNetv2 + AIFI + deformable-attention decoder |

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
| `test_yolov8_arch`   | YOLOv8n architecture: param count (~3.16M), strides, output shapes |
| `test_pt_loader`     | Loads real `yolov8n.pt`, verifies all 355 keys/shapes |
| `test_predict`       | End-to-end inference on `bus.jpg`, ≥1 person + ≥1 bus |
| `test_train_overfit` | Tiny synthetic dataset, training loss decreases ≥ 2× |

## CLI

Two argument styles, both supported:

```
# Ultralytics-style (canonical)
yolocpp task=detect mode=train   model=yolov8n.pt data=path/to/data.yaml epochs=100
yolocpp task=detect mode=val     model=yolov8n.pt data=path/to/data.yaml
yolocpp task=detect mode=predict model=yolov8n.pt source=bus.jpg
yolocpp task=detect mode=export  model=yolov8n.pt format=trt fp16=true
yolocpp mode=benchmark           model=yolov8n.pt source=bus.jpg

# Legacy subcommand-style
yolocpp predict   --weights=yolov8n.pt  --source=bus.jpg --out=out.jpg
yolocpp predict   --weights=yolov8n.trt --source=bus.jpg --out=out.jpg
yolocpp val       --weights=yolov8n.pt  --data=<root>
yolocpp train     --data=<root> [--epochs=100] [--scale=n] [--weights=yolov8n.pt]
yolocpp export    --format=onnx --weights=yolov8n.pt --out=yolov8n.onnx
yolocpp export    --format=trt  --weights=yolov8n.pt --out=yolov8n.trt --fp16
yolocpp benchmark --weights=yolov8n.pt  --source=bus.jpg
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
yolocpp task=classify mode=train   data=DATA model=yolov8n-cls.pt epochs=30
yolocpp task=segment  mode=train   data=DATA model=yolov8n-seg.pt epochs=30
yolocpp task=pose     mode=train   data=DATA model=yolov8n-pose.pt epochs=30
yolocpp task=obb      mode=train   data=DATA model=yolov8n-obb.pt epochs=30
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
  models/yolov8         Conv → C2f → SPPF → Detect (DFL)
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
    yolov8_loss         TAL assigner + CIoU + DFL + BCE
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
  exporter walks our `YoloV8Detect` ModuleList, emits one ONNX node per
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
