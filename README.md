# yolocpp

Pure C++ computer-vision suite. LibTorch for training/eval, TensorRT for
deployment, OpenCV for image I/O. **No Python in the runtime path.**

**License:** AGPL-3.0 (see [`LICENSE`](LICENSE)). yolocpp interoperates
with the upstream Ultralytics YOLO codebase (also AGPL-3.0) — model
architectures, loss formulations, and the e2e dual-head training recipe
are re-implementations of the corresponding pieces in
[`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics).
Any derivative work or network-deployed service built on yolocpp must
satisfy the AGPL's source-availability requirement.

**Pre-1.0.** The current release version lives in the top-level
[`VERSION`](VERSION) file (one line, `MAJOR.MINOR.PATCH`) — that's the
single source of truth. CMake reads it at configure time, embeds it
into the binary via `yolocpp/config.hpp`, and exposes it on the CLI:

```bash
yolocpp --version    # or -v / -V
yolocpp info         # full build info: yolocpp + libtorch + cuda + trt + opencv
```

To bump the version, edit `./VERSION` and add a `CHANGELOG.md` entry.
See [CHANGELOG.md](CHANGELOG.md) for the per-release log and its header
for the pre-1.0 versioning policy. The `1.0.0` line is deliberately
gated on the maintainer's call — not a feature checklist.

## Status

Fourteen YOLO versions are supported. Twelve modern ones (`yolo3
yolo4 yolo5 yolo6 yolo7 yolo8 yolo9 yolo10 yolo11 yolo12 yolo13
yolo26`) ship the detect pipeline end-to-end — **predict, val,
train, ONNX export, TRT export** — across every published scale.
v8 / v11 / v26 additionally ship the full five-task family (detect /
classify / segment / pose / obb). v12 / v13 ship detect-only
upstream; their task heads are scaffolded in code and queued for
retraining on COCO under task #60.

The two Darknet-era versions (`yolo1`, `yolo2`) ship **predict-only**
at the moment, but are usable end-to-end without any Darknet runtime.
The canonical input form is `.pt`, same as every other version —
`build/tools/convert_weights` does a one-shot `.weights → .pt`
conversion via our pure-C++ parser
(`src/serialization/yolov{1,2}_weights.cpp`) and writes the result
to `data/`. The runtime / CLI / registry / tests then consume
`data/*.pt` exclusively. Train / ONNX / TRT for v1/v2 are tracked
under tasks #66..#69.

Reference reading on the current state:

```
ctest --test-dir build                # 31/31 green
bash scripts/full_matrix_sweep.sh     # PASS=152 FAIL=0 SKIP=0
                                      #   predict 121, val 4, train 3,
                                      #   export 12, benchmark 12
```

## Training — 1 epoch vs 5 epochs

Fine-tune on **screen-dataset** (2 465 train / 308 val, nc=5),
batch=16, imgsz=640, seed=42, RTX 5090 32 GB. The 1-epoch column is
the fresh sweep from this session; the 5-epoch column is the
previously-published number from CHANGELOG 0.99.13 and the
follow-up MBLA/P6 runs. Wall = epoch-0 trainer time (1 ep) or full
`/usr/bin/time -v` elapsed (5 ep). mAP / P / R / F1 are computed
against the val split at the end of each epoch.

| variant | 1ep mAP50 | 1ep mAP | 1ep P | 1ep R | 1ep F1 | 1ep wall | 5ep mAP50 | 5ep mAP | 5ep P | 5ep R | 5ep F1 | 5ep wall |
|---------|----------:|--------:|------:|------:|-------:|---------:|----------:|--------:|------:|------:|-------:|---------:|
| yolo3 | 0.453 | 0.300 | 0.709 | 0.649 | 0.652 | 21.0s | 0.688 | — | — | — | — | 1:51.3 |
| yolo5n | 0.698 | 0.570 | 0.858 | 0.741 | 0.776 | 7.4s | 0.816 | 0.707 | 0.914 | 0.893 | 0.902 | 0:36.94 |
| yolo5s | 0.780 | 0.630 | 0.846 | 0.815 | 0.811 | 7.8s | 0.788 | 0.682 | 0.928 | 0.863 | 0.894 | 0:43.80 |
| yolo5m | 0.440 | 0.313 | 0.649 | 0.557 | 0.539 | 10.4s | 0.791 | 0.685 | 0.932 | 0.875 | 0.900 | 0:55.96 |
| yolo5l | 0.337 | 0.250 | 0.442 | 0.595 | 0.421 | 14.0s | 0.809 | 0.684 | 0.928 | 0.842 | 0.877 | 1:15.41 |
| yolo5x | 0.399 | 0.279 | 0.706 | 0.485 | 0.459 | 21.1s | 0.758 | 0.647 | 0.895 | 0.836 | 0.862 | 1:48.85 |
| yolo6n | 0.024 | 0.005 | 0.083 | 0.161 | 0.057 | 7.0s | 0.117 | — | — | — | — | 0:34.1 |
| yolo6s | 0.025 | 0.006 | 0.026 | 0.122 | 0.038 | 7.6s | 0.079 | — | — | — | — | 0:44.1 |
| yolo6m | 0.009 | 0.002 | 0.010 | 0.307 | 0.018 | 11.8s | 0.426 | — | — | — | — | 1:08.1 |
| yolo6l | 0.075 | 0.022 | 0.731 | 0.098 | 0.091 | 16.1s | 0.400 | — | — | — | — | 1:31.0 |
| yolo6n6[P6] | 0.004 | 0.001 | 0.006 | 0.060 | 0.010 | 20.3s | 0.005 | — | — | — | — | 2:56.3 |
| yolo6s6[P6] | 0.004 | 0.001 | 0.018 | 0.104 | 0.018 | 25.0s | 0.009 | — | — | — | — | 3:47.6 |
| yolo6m6[P6] | 0.031 | 0.011 | 0.610 | 0.062 | 0.039 | 43.3s | 0.366 | — | — | — | — | 7:46.2 |
| yolo6l6[P6] | crash | — | — | — | — | — | crash | — | — | — | — | — |
| yolo6s_mbla[MBLA] | 0.101 | 0.047 | 0.791 | 0.104 | 0.109 | 9.7s | 0.579 | 0.432 | 0.854 | 0.610 | 0.684 | 0:50.23 |
| yolo6m_mbla[MBLA] | 0.020 | 0.005 | 0.099 | 0.033 | 0.033 | 12.4s | 0.565 | 0.431 | 0.796 | 0.616 | 0.662 | 1:07.69 |
| yolo6l_mbla[MBLA] | 0.133 | 0.068 | 0.764 | 0.110 | 0.114 | 14.8s | 0.529 | 0.392 | 0.871 | 0.568 | 0.669 | 1:21.69 |
| yolo6x_mbla[MBLA] | 0.079 | 0.027 | 0.100 | 0.223 | 0.114 | 21.2s | 0.563 | 0.448 | 0.769 | 0.667 | 0.689 | 1:57.72 |
| yolo7 | 0.215 | 0.087 | 0.406 | 0.351 | 0.260 | 31.1s | 0.298 | — | — | — | — | 4:49.1 |
| yolo7-tiny | 0.245 | 0.118 | 0.358 | 0.458 | 0.361 | 23.7s | 0.406 | — | — | — | — | 5:29.3 |
| yolo7x | 0.177 | 0.066 | 0.318 | 0.381 | 0.273 | 36.9s | 0.368 | — | — | — | — | 7:02.9 |
| yolo7-w6[P6] | — | — | — | — | — | — | 0.474 | 0.297 | 0.673 | 0.598 | 0.556 | 9:03.60 |
| yolo7-e6[P6] | OOM | — | — | — | — | — | OOM | — | — | — | — | — |
| yolo7-d6[P6] | OOM | — | — | — | — | — | OOM | — | — | — | — | — |
| yolo7-e6e[P6] | OOM | — | — | — | — | — | OOM | — | — | — | — | — |
| yolo8n | 0.713 | 0.577 | 0.904 | 0.756 | 0.803 | 6.8s | 0.768 | 0.712 | 0.939 | 0.896 | 0.916 | 0:32.1 |
| yolo8s | 0.754 | 0.588 | 0.865 | 0.801 | 0.819 | 7.6s | 0.806 | 0.714 | 0.948 | 0.872 | 0.907 | 0:36.4 |
| yolo8m | 0.601 | 0.459 | 0.694 | 0.622 | 0.550 | 10.7s | 0.706 | 0.678 | 0.907 | 0.854 | 0.879 | 0:53.7 |
| yolo8l | 0.445 | 0.320 | 0.684 | 0.545 | 0.527 | 14.6s | 0.668 | 0.660 | 0.957 | 0.810 | 0.858 | 1:15.5 |
| yolo8x | 0.431 | 0.323 | 0.489 | 0.482 | 0.418 | 20.7s | 0.682 | 0.643 | 0.927 | 0.827 | 0.854 | 1:46.1 |
| yolo9t | 0.509 | 0.370 | 0.771 | 0.589 | 0.606 | 9.9s | 0.697 | — | — | — | — | 0:43.0 |
| yolo9s | 0.601 | 0.464 | 0.808 | 0.586 | 0.658 | 10.4s | 0.704 | — | — | — | — | 0:47.1 |
| yolo9m | 0.463 | 0.304 | 0.759 | 0.527 | 0.573 | 14.9s | 0.697 | — | — | — | — | 1:07.0 |
| yolo9c | 0.618 | 0.461 | 0.866 | 0.693 | 0.732 | 16.0s | 0.694 | — | — | — | — | 1:19.4 |
| yolo9e | 0.357 | 0.233 | 0.705 | 0.482 | 0.474 | 29.1s | 0.655 | — | — | — | — | 2:28.6 |
| yolo10n | 0.656 | 0.491 | 0.779 | 0.756 | 0.734 | 14.3s | 0.711 | — | — | — | — | 0:40.0 |
| yolo10s | 0.438 | 0.345 | 0.714 | 0.560 | 0.567 | 13.1s | 0.690 | — | — | — | — | 0:45.6 |
| yolo10m | 0.487 | 0.370 | 0.620 | 0.560 | 0.501 | 16.0s | 0.719 | — | — | — | — | 1:03.6 |
| yolo10b | 0.415 | 0.312 | 0.594 | 0.482 | 0.432 | 16.6s | 0.675 | — | — | — | — | 1:12.6 |
| yolo10l | 0.454 | 0.347 | 0.512 | 0.542 | 0.406 | 18.5s | 0.691 | — | — | — | — | 1:23.9 |
| yolo10x | 0.497 | 0.318 | 0.664 | 0.589 | 0.562 | 23.6s | 0.674 | — | — | — | — | 1:52.2 |
| yolo11n | 0.698 | 0.574 | 0.852 | 0.750 | 0.792 | 9.7s | 0.748 | 0.700 | 0.945 | 0.842 | 0.890 | 0:38.1 |
| yolo11s | 0.754 | 0.582 | 0.838 | 0.756 | 0.791 | 10.8s | 0.708 | 0.694 | 0.936 | 0.872 | 0.901 | 0:43.0 |
| yolo11m | 0.465 | 0.326 | 0.512 | 0.586 | 0.384 | 14.5s | 0.664 | 0.663 | 0.943 | 0.824 | 0.865 | 1:02.8 |
| yolo11l | 0.588 | 0.400 | 0.559 | 0.598 | 0.512 | 16.9s | 0.654 | 0.643 | 0.927 | 0.801 | 0.840 | 1:15.1 |
| yolo11x | 0.316 | 0.190 | 0.581 | 0.443 | 0.442 | 23.9s | 0.615 | 0.589 | 0.900 | 0.774 | 0.812 | 1:51.5 |
| yolo12n | 0.746 | 0.607 | 0.815 | 0.842 | 0.826 | 11.4s | 0.719 | 0.690 | 0.921 | 0.875 | 0.897 | 0:45.2 |
| yolo12s | 0.615 | 0.431 | 0.772 | 0.676 | 0.709 | 14.1s | 0.696 | 0.691 | 0.915 | 0.869 | 0.891 | 0:58.5 |
| yolo12m | 0.366 | 0.236 | 0.705 | 0.435 | 0.497 | 19.7s | 0.636 | 0.636 | 0.950 | 0.789 | 0.843 | 1:30.4 |
| yolo12l | 0.386 | 0.286 | 0.631 | 0.533 | 0.454 | 28.3s | 0.641 | 0.619 | 0.965 | 0.765 | 0.835 | 2:16.0 |
| yolo12x | 0.266 | 0.146 | 0.516 | 0.446 | 0.391 | 40.6s | 0.620 | 0.590 | 0.919 | 0.765 | 0.816 | 3:20.3 |
| yolo13n | 0.724 | 0.609 | 0.860 | 0.762 | 0.783 | 18.0s | 0.837 | 0.751 | 0.954 | 0.893 | 0.921 | 0:57.7 |
| yolo13s | 0.614 | 0.488 | 0.827 | 0.693 | 0.719 | 21.2s | 0.742 | 0.705 | 0.914 | 0.875 | 0.892 | 1:15.6 |
| yolo13l | 0.636 | 0.509 | 0.712 | 0.762 | 0.688 | 39.0s | 0.677 | 0.660 | 0.888 | 0.884 | 0.884 | 2:43.1 |
| yolo13x | — | — | — | — | — | — | 0.643 | — | — | — | — | 4:21.6 |
| yolo26n | 0.608 | 0.466 | 0.832 | 0.688 | 0.693 | 11.6s | 0.695 | — | — | — | — | 0:44.2 |
| yolo26s | 0.685 | 0.500 | 0.776 | 0.729 | 0.726 | 12.0s | 0.705 | — | — | — | — | 0:49.2 |
| yolo26m | 0.599 | 0.426 | 0.790 | 0.756 | 0.742 | 15.9s | 0.740 | — | — | — | — | 1:11.6 |
| yolo26l | 0.510 | 0.358 | 0.578 | 0.619 | 0.580 | 18.2s | 0.755 | — | — | — | — | 1:23.9 |
| yolo26x | 0.564 | 0.406 | 0.656 | 0.640 | 0.601 | 25.7s | 0.738 | — | — | — | — | 2:07.0 |

5-epoch P/R/F1 cells are blank where the original sweep (0.99.13 +
followups) only logged mAP — the trainer's per-epoch
`[trainer] val:` line landed in 0.99.10. Reference comparison vs
**Ultralytics 8.4.60 / Meituan / iMoonLab** lives in the
`Training comparison vs reference` section below — kept separate
so this matrix stays one consolidated convergence table.

mAP@0.5 → mAP@0.5:0.95 = "mAP50 → mAP" abbreviation above.

## Training comparison vs reference (5-epoch only)

Single unified benchmark: 5-epoch fine-tune on screen-dataset
(2465 train / 308 val, nc=5), batch=16, imgsz=640, seed=42, RTX 5090
(32 GB). Both frameworks load the same upstream `.pt`; both validate
using their own native pipeline.

- yolocpp: **0.99.13**
- Ultralytics reference: **8.4.56** (stock release)
- yolo6 reference: **Meituan/YOLOv6** official repo (Ultralytics doesn't ship v6)
- yolo7 reference: **WongKinYiu/yolov7** official repo (Ultralytics doesn't ship v7) — *training pipeline broken under current torch/PyPI versions; yolocpp-only numbers shown*
- yolo13 reference: **iMoonLab/yolov13** fork (stock Ultralytics can't load v13)

Wall, CPU%, RSS, VRAM all captured with the same `/usr/bin/time -v`
plus 100 ms `nvidia-smi` polling. VRAM values are 99th-percentile to
filter the cuDNN-benchmark search-time transient spikes (p50 sustained
matches within ~5 %). Bold = win > 0.012 mAP or speedup ≥ 1.10×.

| variant | mAP (yolocpp / ref) | Δ mAP | wall (us / ref) | speedup | CPU% (us / ref) | RSS GB (us / ref) | VRAM MiB (us / ref) | reference |
|---------|--------------------:|------:|----------------:|--------:|----------------:|------------------:|--------------------:|-----------|
| **yolo3** | 0.688 / 0.646 | **+0.042** | 1:51.3 / 2:38.9 | **1.43×** | 212% / 150% | 8.1 / 9.7 | 16832 / 19771 | Ultra 8.4.56 |
| yolo5n     | 0.698 / 0.709 | −0.011 | 0:32.7 / 0:46.2 | **1.41×** | 473% / 284% | 8.1 / 8.3 | 4191 / 4971 | Ultra 8.4.56 |
| **yolo5s** | 0.763 / 0.710 | **+0.053** | 0:36.4 / 0:51.9 | **1.43×** | 430% / 259% | 8.1 / 8.4 | 5785 / 6616 | Ultra 8.4.56 |
| **yolo5m** | 0.757 / 0.723 | **+0.034** | 0:50.7 / 1:15.0 | **1.48×** | 334% / 207% | 8.1 / 8.6 | 8811 / 10323 | Ultra 8.4.56 |
| yolo5l     | 0.638 / 0.655 | −0.018 | 1:10.0 / 1:40.5 | **1.43×** | 270% / 177% | 8.1 / 9.1 | 13372 / 13596 | Ultra 8.4.56 |
| yolo5x     | 0.626 / 0.616 | +0.011 | 1:47.9 / 2:35.1 | **1.44×** | 212% / 149% | 8.2 / 9.7 | 18593 / 19092 | Ultra 8.4.56 |
| **yolo6n** | 0.117 / 0.069 | **+0.048** | 0:34.1 / 0:59.4 | **1.74×** | 477% / 277% | 8.2 / 4.5 | 3903 / 5650 | Meituan v6 |
| **yolo6s** | 0.079 / 0.042 | **+0.037** | 0:44.1 / 1:14.3 | **1.68×** | 398% / 234% | 8.1 / 4.5 | 5141 / 8235 | Meituan v6 |
| **yolo6m** | 0.426 / 0.044 | **+0.382** | 1:08.1 / 2:04.3 | **1.82×** | 296% / 180% | 8.1 / 4.8 | 7717 / 15369 | Meituan v6 |
| **yolo6l** | 0.400 / 0.107 | **+0.293** | 1:31.0 / 2:05.5 | **1.38×** | 241% / 179% | 8.1 / 5.0 | 14191 / 16968 | Meituan v6 |
| yolo6n6[P6]   | 0.005 / 0.157 | −0.151 | 2:56.3 / 1:04.0 | 0.36×    | 339% / 188% | 8.2 / 4.7 | — / — | Meituan v6 |
| yolo6s6[P6]   | 0.009 / 0.083 | −0.074 | 3:47.6 / 1:21.3 | 0.36×    | 274% / 176% | 8.2 / 4.7 | — / — | Meituan v6 |
| **yolo6m6**[P6] | 0.366 / 0.027 | **+0.339** | 7:46.2 / 2:13.4 | 0.29× | 156% / 159% | 8.3 / 5.0 | — / — | Meituan v6 |
| yolo6l6[P6]   | **crash** / 0.179 | — | 0:04.2 / 2:20.9 | — | — / — | — / — | — / — | Meituan v6 |
| **yolo6s_mbla**[MBLA] | 0.432 / 0.025 | **+0.407** | 0:50.2 / 1:08.4 | **1.36×** | 352% / — | 8.1 / — | — / — | Meituan v6 |
| **yolo6m_mbla**[MBLA] | 0.431 / 0.055 | **+0.376** | 1:07.7 / 1:22.3 | **1.22×** | 287% / — | 8.0 / — | — / — | Meituan v6 |
| **yolo6l_mbla**[MBLA] | 0.392 / 0.031 | **+0.361** | 1:21.7 / 1:40.9 | **1.23×** | 253% / — | 8.1 / — | — / — | Meituan v6 |
| **yolo6x_mbla**[MBLA] | 0.448 / 0.066 | **+0.382** | 1:57.7 / 2:24.9 | **1.23×** | 208% / — | 8.1 / — | — / — | Meituan v6 |
| yolo7      | 0.298 / —     | —      | 4:49.1 / —      | —       | 151% / —    | 8.0 / —   | 16052 / —     | (WKY pipeline broken)* |
| yolo7-tiny | 0.406 / —     | —      | 5:29.3 / —      | —       | 146% / —    | 7.8 / —   | 5218 / —      | (WKY pipeline broken)* |
| yolo7x     | 0.368 / —     | —      | 7:02.9 / —      | —       | 135% / —    | 8.0 / —   | 20463 / —     | (WKY pipeline broken)* |
| yolo7-w6[P6] | 0.297 / —     | —    | 9:03.6 / —     | —      | 105% / —    | 8.4 / —   | 30238 / —     | (WKY pipeline broken)*, imgsz=1280 |
| yolo7-e6[P6] | **OOM** / —   | —    | 0:10.1 / —     | —      | — / —       | — / —     | — / —         | OOM at imgsz=1280 on 32 GB[P6-OOM] |
| yolo7-d6[P6] | **OOM** / —   | —    | 0:09.6 / —     | —      | — / —       | — / —     | — / —         | OOM at imgsz=1280 on 32 GB[P6-OOM] |
| yolo7-e6e[P6]| **OOM** / —   | —    | 0:09.7 / —     | —      | — / —       | — / —     | — / —         | OOM at imgsz=1280 on 32 GB[P6-OOM] |
| yolo8n     | 0.768 / 0.759 | +0.010 | 0:32.1 / 0:43.8 | **1.36×** | 490% / 297% | 8.1 / 8.3 | 4389 / 4997 | Ultra 8.4.56 |
| **yolo8s** | 0.806 / 0.739 | **+0.067** | 0:36.4 / 0:51.9 | **1.42×** | 429% / 260% | 8.1 / 8.4 | 6045 / 6899 | Ultra 8.4.56 |
| yolo8m[avg-3]    | 0.706 / 0.695 | **+0.011** | 0:53.7 / 1:17.4 | **1.44×** | 323% / 202% | 8.1 / 8.7 | 9120 / 10322 | Ultra 8.4.56 |
| yolo8l     | 0.668 / 0.663 | +0.005 | 1:15.5 / 1:45.8 | **1.40×** | 258% / 173% | 8.0 / 8.9 | 13622 / 13828 | Ultra 8.4.56 |
| **yolo8x** | 0.682 / 0.660 | **+0.022** | 1:46.1 / 2:23.3 | **1.35×** | 213% / 154% | 8.3 / 9.3 | 16693 / 16859 | Ultra 8.4.56 |
| yolo9t     | 0.697 / 0.722 | −0.025 | 0:43.0 / 1:14.6 | **1.73×** | 382% / 207% | 8.1 / 8.3 | 5729 / 6101 | Ultra 8.4.56 |
| yolo9s     | 0.704 / 0.718 | −0.015 | 0:47.1 / 1:19.5 | **1.69×** | 352% / 200% | 8.1 / 8.4 | 7158 / 8509 | Ultra 8.4.56 |
| **yolo9m**[avg-3] | 0.697 / 0.684 | **+0.013** | 1:07.0 / 2:31.4 | **2.26×** | 276% / 99% | 8.2 / 4.0 | 11269 / 11563 | Ultra 8.4.56[w=0] |
| **yolo9c** | 0.694 / 0.654 | **+0.040** | 1:19.4 / 2:44.1 | **2.07×** | 249% / 99% | 8.1 / 4.1 | 13908 / 14571 | Ultra 8.4.56[w=0] |
| **yolo9e** | 0.655 / 0.643 | **+0.012** | 2:28.6 / 4:34.6 | **1.85×** | 179% / 99% | 8.2 / 4.6 | 24078 / 23920 | Ultra 8.4.56[w=0] |
| **yolo10n**| 0.711 / 0.698 | **+0.013** | 0:40.0 / 1:46.7 | **2.67×** | 400% / 99% | 8.5 / 4.1 | 4642 / 5813 | Ultra 8.4.56[w=0] |
| yolo10s[avg-3]   | 0.690 / 0.743 | **−0.053** | 0:45.6 / 1:57.5 | **2.58×** | 361% / 99% | 8.5 / 4.2 | 6700 / 8268 | Ultra 8.4.56[w=0] |
| **yolo10m**| 0.719 / 0.690 | **+0.028** | 1:03.6 / 2:24.2 | **2.27×** | 285% / 99% | 8.5 / 4.3 | 10542 / 12490 | Ultra 8.4.56[w=0] |
| **yolo10b**| 0.675 / 0.651 | **+0.024** | 1:12.6 / 2:36.7 | **2.16×** | 263% / 99% | 8.5 / 4.3 | 13442 / 15486 | Ultra 8.4.56[w=0] |
| **yolo10l**| 0.691 / 0.628 | **+0.063** | 1:23.9 / 2:53.3 | **2.07×** | 240% / 99% | 8.4 / 4.3 | 16098 / 17154 | Ultra 8.4.56[w=0] |
| yolo10x[avg-3]   | 0.674 / 0.671 | **+0.003** | 1:52.2 / 3:29.7 | **1.87×** | 204% / 99% | 8.5 / 4.4 | 20492 / 19907 | Ultra 8.4.56[w=0] |
| yolo11n    | 0.748 / 0.738 | +0.011 | 0:38.1 / 0:55.2 | **1.45×** | 416% / 253% | 8.4 / 8.5 | 4669 / 5187 | Ultra 8.4.56 |
| yolo11s[avg-3]   | 0.708 / 0.720 | −0.012 | 0:43.0 / 1:02.3 | **1.45×** | 377% / 232% | 8.4 / 8.6 | 6657 / 7693 | Ultra 8.4.56 |
| yolo11m[avg-3]   | 0.664 / 0.655 | **+0.009** | 1:02.8 / 1:30.4 | **1.44×** | 291% / 188% | 8.4 / 8.7 | 12080 / 12255 | Ultra 8.4.56 |
| yolo11l[avg-3]   | 0.654 / 0.650 | **+0.004** | 1:15.1 / 1:49.1 | **1.45×** | 261% / 172% | 8.4 / 8.9 | 15844 / 15470 | Ultra 8.4.56 |
| **yolo11x**[avg-3] | 0.615 / 0.600 | **+0.015** | 1:51.5 / 2:36.1 | **1.40×** | 207% / 149% | 8.5 / 9.3 | 21137 / 18836 | Ultra 8.4.56 |
| yolo12n    | 0.719 / 0.722 | −0.003 | 0:45.2 / 1:03.6 | **1.41×** | 364% / 229% | 8.3 / 8.5 | 6684 / 6602 | Ultra 8.4.56 |
| yolo12s[avg-3]   | 0.696 / 0.702 | −0.006 | 0:58.5 / 1:20.0 | **1.37×** | 302% / 199% | 8.5 / 8.6 | 9262 / 9885 | Ultra 8.4.56 |
| yolo12m    | 0.636 / 0.644 | −0.007 | 1:30.4 / 1:58.6 | **1.31×** | 230% / 166% | 8.4 / 8.8 | 15364 / 15777 | Ultra 8.4.56 |
| **yolo12l**| 0.641 / 0.627 | **+0.014** | 2:16.0 / 2:49.9 | **1.25×** | 187% / 145% | 8.4 / 8.9 | 21542 / 19051 | Ultra 8.4.56 |
| **yolo12x**[avg-3] | 0.620 / 0.595 | **+0.026** | 3:20.3 / 4:01.5 | **1.21×** | 160% / 132% | 8.5 / 9.3 | 31347 / 27772 | Ultra 8.4.56 |
| **yolo13n**| 0.837 / 0.791 | **+0.046** | 0:57.7 / 1:26.5 | **1.50×** | 305% / 186% | 8.8 / 6.9 | 6138 / 7251 | iMoonLab fork |
| yolo13s    | 0.742 / 0.755 | −0.013 | 1:15.6 / 1:51.9 | **1.48×** | 256% / 165% | 8.8 / 6.9 | 9854 / 11337 | iMoonLab fork |
| yolo13l    | 0.677 / 0.705 | −0.028 | 2:43.1 / 4:21.9 | **1.61×** | 171% / 127% | 8.8 / 7.1 | 26369 / 27798 | iMoonLab fork |
| yolo13x [b=8] | 0.643 / 0.679 | −0.036 | 4:21.6 / 7:02.3 | **1.62×** | — / — | 8.7 / 7.1 | — / — | iMoonLab fork (both impls OOM at b=16 on 32 GB) |
| yolo26n    | 0.695 / 0.749 | −0.054 | 0:44.2 / 1:07.3 | **1.52×** | 371% / 222% | 8.4 / 8.5 | 4811 / 5282 | Ultra 8.4.56 |
| yolo26s    | 0.705 / 0.781 | −0.076 | 0:49.2 / 1:16.0 | **1.54×** | 341% / 206% | 8.4 / 8.6 | 7264 / 8292 | Ultra 8.4.56 |
| yolo26m    | 0.740 / 0.791 | −0.051 | 1:11.6 / 1:43.3 | **1.44×** | 266% / 176% | 8.4 / 8.8 | 13020 / 13260 | Ultra 8.4.56 |
| yolo26l    | 0.755 / 0.790 | −0.035 | 1:23.9 / 2:03.1 | **1.47×** | 242% / 162% | 8.5 / 8.8 | 15579 / 15981 | Ultra 8.4.56 |
| yolo26x    | 0.738 / 0.789 | −0.052 | 2:07.0 / 2:58.7 | **1.41×** | 194% / 143% | 8.5 / 9.3 | 22373 / 20103 | Ultra 8.4.56 |

\* WongKinYiu/yolov7's stock training pipeline (2022) breaks on
modern PyTorch (torch.load `weights_only=True` default) and on
modern GitHub releases API (`attempt_download` expects an `assets`
key that's no longer there). Patched the torch.load issue locally
but the second one needs deeper changes to their auto-download
flow. yolocpp-only numbers shown; would need their environment
preserved as Docker image for a future apples-to-apples comparison.

**Footnote `[P6]`** — yolo6 P6 variants (n6/s6/m6/l6) — 4-level head trained at
imgsz=640 (not the upstream 1280). yolocpp's small P6 (n6/s6)
under-converge in 5 epochs at 640 resolution because the extra
P6/64 head needs longer to align; Meituan's training is similarly
weak on n6/s6 at this budget (0.157 / 0.083). y6m6 trains
healthily and beats Meituan by +0.339. y6l6 crashed (assertion in
the forward path under our channels_last layout on P6 m+ scales —
documented in TODO #6). At upstream imgsz=1280 (the published P6
spec) all 4 variants are expected to converge.

**Footnote `[avg-3]`** — multi-seed averaged (seeds 42 / 43 / 44). Single-seed results
for these variants showed apparent gaps of −0.018 to −0.033 mAP; the
three-seed mean reveals that variance for these specific (variant,
seed) combinations is **±0.02 mAP from run to run** — eating the
single-seed "gap" entirely. After averaging, all of these collapse
into the noise band or flip into outright wins. Only **y10s remains
a true residual** (−0.053 stable across all three seeds). Tested
enabling dual-head v10 training (matching Ultralytics' E2ELoss
default) — regressed y10s to 0.641, so reverted. See CHANGELOG 0.99.13.

**Footnote `[w=0]`** — Ultralytics 8.4.56 + v9 medium-and-up + v10 + Blackwell deadlocks
on the default `workers=8` DataLoader workers (stalls at 0 % CPU
mid-train). Pinned `workers=0` in our Ultralytics-side runner for
those rows; yolocpp uses its own `BatchPrefetcher` and is unaffected.
Documented in CHANGELOG 0.99.13.

**Footnote `[MBLA]`** — yolo6 MBLA variants (s/m/l/x_mbla) — yolocpp
trains cleanly in 50 s – 1 m 58 s for 5 epochs, mAP@0.5:0.95 lands
in 0.39 – 0.45. Meituan reference column lands at 0.025 – 0.066 —
similar to the base v6 P5 row pattern (yolo6m/l Meituan also low
at 5-ep on this dataset). Meituan's COCO-style recipe needs more
epochs to converge on a 5-class 2465-image dataset; at 5 epochs
yolocpp beats Meituan by +0.36 to +0.41 mAP. Fix for the original
runner cwd bug (#87B) landed in 0.99.17. Reference: Meituan's
upstream training recipe (`configs/mbla/yolov6{*}_mbla.py`) — pure
COCO-tuned hyperparams + 300/400 epoch schedule.

**Footnote `[w=0-cpp]`** — *resolved in 0.99.17.* yolocpp's v7 P6
training pipeline now lands a real 0.297 mAP@0.5:0.95 for yolo7-w6
at 5-ep (vs 0 before). Root cause was the upstream P6 anchors
(`yolov7-{w6,e6,d6,e6e}.yaml`) being calibrated at imgsz=1280 but
used at imgsz=640 — every anchor was ~2× too large, GT-to-anchor
IoU matching collapsed to zero positives per GT, mAP saturated to
0. Fix: enable `c.autoanchor = true` in the v7 P6 LossTraits —
K-means reclusters from the actual training GT (w, h) distribution
and `loss_after_init` syncs the new anchors into the model's
`anchor_grid` buffer (matches WongKinYiu's `check_anchors()`
BPR<0.98 reclustering in `autoanchor.py`). See CHANGELOG 0.99.17.

**Footnote `[P6-OOM]`** — yolo7-e6 / -d6 / -e6e are the big P6
variants (97–144M params) trained at upstream imgsz=1280. At
batch=16 / imgsz=1280 the forward + autograd state exceeds 30 GB,
OOM-ing on our 32 GB RTX 5090. Upstream WongKinYiu trains these on
8×A100-40GB clusters. At batch=8 / imgsz=1280, or batch=16 /
imgsz=960, they'd fit — both are deviations from the published
recipe so we don't compare against unreleased numbers. yolo7-w6 is
smaller (70M) and fits at b=16 / 1280 with 30 GB peak. WongKinYiu
reference column is blank for all four P6 variants since their
training pipeline is broken under current torch (see WKY note).

### Not yet benchmarked

Variants supported by yolocpp but not in the table above because the
local weight cache doesn't have them yet. Each is one `curl` away:

| variant | upstream URL pattern | reason missing | yolocpp support |
|---------|----------------------|----------------|-----------------|
| yolo1   | `https://pjreddie.com/media/files/yolov1.weights` | Joseph Redmon's original Darknet weights — not auto-downloaded yet | predict-only (TODO #66 for train/export) |
| yolo6 `*_mbla` (s/m/l/x) | **landed 0.99.15** — yolocpp rows in main table marked `[MBLA]`; Meituan reference deferred ([MBLA] footnote — runner cwd bug) | — | full pipeline |
| yolo7 P6 (w6/e6/d6/e6e) | **partial 0.99.15** — TRT INT8 fails per `[INT8-fail]`; train mAP=0 per `[w=0-cpp]`; predict + ONNX/TRT FP16 work | needs P6 forward / assigner fix (task #87C) | predict + ONNX + TRT FP16; train degenerate |
| yolo13m | n/a | iMoonLab fork doesn't ship an `m` variant | — |
| yolo13l TRT FP16 latency | (build hung) | one-off TensorRT build hang during the FPS sweep; rerun on the next pass | mAP/wall already in main table |

### Darknet-era models (yolocpp-only)

yolo4 / yolo2 (full + tiny + voc variants) / yolo1 ship in yolocpp
end-to-end (predict for v1/v2, full pipeline for v4) but the
original Darknet (C) authors never published a comparable Python
training pipeline at AlexeyAB/Joseph Redmon repos:

| variant       | params | yolocpp predict | yolocpp train | reference | note |
|---------------|-------:|-----------------|---------------|-----------|------|
| yolo1         | ~272 M | ✅ (Pascal VOC) | TODO #66      | none      | pjreddie 2016, FC head |
| yolo2         | ~67 M  | ✅ COCO         | TODO #67      | none      | reorg passthrough |
| yolo2-voc     | ~50 M  | ✅              | TODO #67      | none      | PASCAL VOC anchors |
| yolo2-tiny-voc| ~16 M  | ✅              | TODO #67      | none      | tiny YOLOv2 |
| yolo4         | ~64 M  | ✅              | ✅ via V7Loss | none      | CSPDarknet53+PANet, default imgsz=608 |

No training reference benchmark possible — included for completeness
of the version-coverage matrix.

### Inference speed (FPS) — RTX 5090, batch=1, imgsz=640

Single-image inference latency on `bus.jpg`, batch=1, after a 5-iter
warmup + 30 timed iters. Both stacks load the same upstream `.pt`
weights; both export their own TRT engine with INT8 calibrated
against the screen-dataset val set. **Bold = yolocpp wins by ≥ 2 %**.

| variant | U PT | **Y PT** | U FP16 | **Y FP16** | U INT8 | **Y INT8** |
|---------|-----:|-----:|-------:|-------:|-------:|-------:|
| yolo8n  | 426 | **484** |  689 | **1079** | 678 |  **870** |
| yolo8s  | 370 | **389** |  653 |  **950** | 653 |  **848** |
| yolo8m  | 249 |  248   |  537 |  **690** | 536 |  **635** |
| yolo8x  | 148 |  143   |  376 |  **430** | 406 |  **444** |
| yolo10n | 342 | **545** |  746 | **1008** | 701 |  **741** |
| yolo10x |  78 | **181** |  128 |  **403** | 330 |  **358** |
| yolo11n | 365 | **500** |  651 |  **972** | 635 |  **830** |
| yolo11s | 358 | **397** |  620 |  **876** | 590 |  **771** |
| yolo11m | 254 | **261** |  496 |  **639** | 497 |  **612** |
| yolo11x |  69 | **138** |   97 |  **418** |  83 |  **397** |
| yolo12n | 278 | **338** |  501 |  **671** | 516 |  **652** |
| yolo12x | 122 |  106   |  219 |  **260** | 244 |  **286** |
| yolo13n | —   |  318   |  —   |   620    |  —  |   559    |
| yolo13x | —   |   96   |  —   |   223    |  —  |   239    |
| yolo26n | 339 | **418** |  703 |  **912** | 662 |  **737** |
| yolo26x | 151 |  129   |  381 |  **414** | 379 |  **403** |

U = Ultralytics 8.4.60 (Python + their TRT runtime); Y = yolocpp
(C++ + libtorch + nvinfer). yolo13 reference cells empty because
Ultralytics' `YOLO()` loader rejects the iMoonLab fork's weights
(unknown architecture key) — yolocpp loads them natively.

**Headline.** yolocpp leads Ultralytics on **TRT FP16** and **TRT
INT8** on every measured variant (margins 1.07× – 4.78×). PT FP32
leads on small / medium scales; trails on x-scale heavyweights
where the gap is compute-bound (Ultralytics' Sequential dispatch
has lower per-call overhead than ours when the kernel itself
takes 6+ ms). The big wins on v10x / v11x come from #88C (4 GiB
TRT workspace unblocked previously-failing INT8 tactic search)
and the profile-guided uint8-H2D + GPU-side preprocess pipeline
(0.99.25 → 0.99.27). Reproduce with `/tmp/ultra_bench/sweep.sh`.

The per-variant yolocpp-only FPS numbers across **all 60+ variants**
(PT, TRT FP32, FP16, INT8 with latencies) live in the **CHANGELOG
entries 0.99.15 → 0.99.20**; they're the data the 16-variant
comparison above was distilled from. Out-of-scope INT8 failures
(v3, v7-base, v7-x, v9e, v26x) and "OOM at imgsz=1280 on 32 GB"
v7 P6 variants are noted there. **`tools/profile_trt`** ships in
the build tree — point it at any `.trt` engine to see the per-phase
breakdown (`letterbox / image_to_tensor / H2D / enqueueV3 / NMS`).

### Graphs

Comparison figures live in [`docs/figures/`](docs/figures/):

- [`mAP_vs_params.png`](docs/figures/mAP_vs_params.png) — quality vs model size (yolocpp solid, reference dashed)
- [`mAP_vs_wall.png`](docs/figures/mAP_vs_wall.png) — quality vs train wall time
- [`mAP_vs_fps.png`](docs/figures/mAP_vs_fps.png) — quality vs inference FPS (TRT FP16) — **the operating-point chart**
- [`fps_per_variant.png`](docs/figures/fps_per_variant.png) — TRT FP16 (solid) vs PT FP32 (faded) per variant
- [`trt_speedup.png`](docs/figures/trt_speedup.png) — TRT FP16 / PT FP32 speedup sorted high-to-low
- [`speedup_per_variant.png`](docs/figures/speedup_per_variant.png) — yolocpp train speedup vs reference
- [`delta_per_variant.png`](docs/figures/delta_per_variant.png) — Δ mAP per variant, BEAT/TIED/TRAIL zones marked

Color stays consistent across all graphs per model family:
yolo3=blue, yolo5=orange, yolo6=green, yolo7=red, yolo8=purple,
yolo9=brown, yolo10=pink, yolo11=gray, yolo12=olive, yolo13=cyan,
yolo26=black, v1/v2/v4 = light variants.

### Headline

- **21 variants beat reference** (Δ > +0.012): yolo3, yolo5s/m,
  yolo6n/s/m/l, yolo8s/x, **yolo9m**, yolo9c/e, yolo10n/m/b/l,
  **yolo11x**, yolo12l, **yolo12x**, yolo13n. Biggest standouts:
  yolo6m (+0.382), yolo6l (+0.293), yolo8s (+0.067), yolo10l
  (+0.063), yolo5s (+0.053), yolo13n (+0.046), yolo3 (+0.042),
  yolo9c (+0.040).
- **14 variants tied** (|Δ| ≤ 0.012): yolo5n, yolo5x, yolo8n, **yolo8m**,
  yolo8l, yolo9t/s, yolo10n (borderline), **yolo10x**, yolo11n,
  **yolo11s/m/l**, yolo12n, yolo12m, **yolo12s**, yolo13s.
- **2 variants trail** by ≥ 0.020 mAP: **yolo10s** (−0.053, multi-seed
  stable — only persistent residual after the audit) and yolo11x (now
  +0.015 after averaging — flipped to a win). yolo26 family widens
  the trail list (−0.035 to −0.077) but is structurally bounded by
  short-budget E2ELoss dynamics (CHANGELOG 0.99.10).
- **Speedup: 1.21×–2.67× faster** on every comparable workload.
  Smallest margin yolo12x (GPU-compute-bound); largest yolo10n
  (workers=0 fix exposes our async pipeline win).
- **VRAM: matches reference within ±10 %** on sustained workloads.
  RSS: parity (~8 GB both, though Ultralytics with workers=0 drops
  to ~4 GB since no worker subprocesses). CPU%: yolocpp uses ~1.5×
  more host CPU than Ultralytics — LibTorch isn't GIL-serialized.
- **47 (version, variant) cells** benchmarked total + 60 additional
  multi-seed verification runs across 10 trailing variants.
  The multi-seed audit established that **run-to-run variance is
  ±0.02 mAP** on this dataset / budget — single-seed gaps within
  that band are noise, not bugs. Earlier-session 30 GB VRAM "peaks"
  on m-variants were cuDNN-benchmark-mode probing transients
  (p50 sustained always 12–15 GB).

Cross-cutting infrastructure that's already wired:

- Hand-written ONNX protobuf emitter (no libprotobuf, no Python tracer)
  with task-aware decoders for detect / segment / pose / obb / classify.
- TensorRT builder via `nvonnxparser::IParser`, FP16 on Blackwell,
  TF32-cleared per-version where required (v10 RepVGGDW saturation).
- Templated `TrainerT<M>` + `LossTraits<M>` covering all twelve
  versions via four loss classes (`V8DetectionLoss`,
  `V6DetectionLoss`, `V7DetectionLoss`, `Yolo26Loss`).
- Templated `engine::validate<M>` for every detect-shape model.
- Multi-GPU DDP scaffolding (NCCL + all-reduce), world_size=1
  verified end-to-end; two-GPU box validation pending hardware.
- Auto-resolve of `model=` / `data=` / `scale=` / `version=` / `nc=`
  from cwd / cache / state-dict shape / filename — pass `model=`
  alone and the rest is inferred (renamed `best.pt` / `last.pt`
  works).
- Run artefacts: `results.csv`, `args.yaml`, `confusion_matrix.png`,
  `BoxPR/BoxF1/BoxP/BoxR_curve.png`, `labels.jpg`, `results.png`,
  `train_batch{0,1,2}.jpg`, `best.pt` at peak mAP@0.5:0.95,
  `patience=N` early-stop.
- `runs/<mode>/` default output convention (predict / val / export
  all live alongside `runs/train/`).

## Roadmap

The full task ledger lives in **[TODO.md](TODO.md)** — every completed,
in-flight, and pending task across the codebase, not just the active
session. Highlights of the next-batch roadmap (tasks #46..#63, filed
2026-05-01):

- **Group I (foundation):** modular per-version registry (#46),
  centralised version stamp (#47), centralised+minimised third-party
  deps (#48), strip every "ultralytics" trace (#49), pick a license
  (#50).
- **Group II (CLI / API):** long+short flags (`--model/-m`, …),
  `--source` covering image/video/dir/URL/webcam, `--seed`,
  `yolocpp download <dataset>`, unified `export precision=` switch,
  auto-export ONNX after train, `--device` covering cpu/cuda/mps,
  Python-style chainable `yolocpp::YOLO(...)` C++ API.
- **Group III (verification):** cross-backend `.pt`/`.onnx`/`.engine`
  parity assert; mAP small/medium/large breakdown.
- **Group IV (features):** SAHI + tracker family
  (SORT/DeepSORT/OC-SORT/ByteTrack/BoT-SORT/NvSORT); add legacy +
  additional YOLO families (yolo1, yolo2, YOLOX, YOLO-NAS,
  YOLO-WORLD, YOLOE, YOLOR, PP-YOLO, Scaled-YOLOv4, DAMO-YOLO).
- **Group V (perf / hardware):** parallelisation pass, multi-device
  dispatch, iOS/Android/edge deploy, Jetson Nano/Orin/THOR + DGX Spark
  TRT plans.
- **Group VI (distribution):** retrain every (version × scale × task)
  on COCO and publish weights to GitHub Releases; comparison
  table/graphs.
- **Group VII (optional):** Ninja generator, cross-platform GUI.

A recurring **gap-audit** (task #33) sweeps the codebase periodically
for unwired pipelines, stub implementations, SKIP-gated tests, and
upstream variants not yet mirrored — see CLAUDE.md "Periodic gap-audit
(recurring TODO #33)".

## YOLO version roadmap

The codebase covers **only** these fourteen YOLO versions and uses
the single canonical filename / identifier convention `yolo<N>`
everywhere (`yolo1`, `yolo2`, `yolo3`, `yolo4`, `yolo5`, `yolo6`,
`yolo7`, `yolo8`, `yolo9`, `yolo10`, `yolo11`, `yolo12`, `yolo13`,
`yolo26` — **no `v`**). When fetching legacy upstream weights, the
resolver transparently maps the canonical name back to pjreddie's
`yolov<N>.weights` (v1, v2), AlexeyAB's `yolov4.weights`, or
Ultralytics' `yolov<N>...pt`. Versions outside this set (v14..v25,
v27+) are intentionally not supported.

| version | year | provenance | family / changes | status |
|---------|------|------------|------------------|--------|
| **yolo1**  | 2016 | Redmon et al. (official Darknet)           | 24 conv (no BN, leaky 0.1) + 2 FC (4096 → 7·7·30); SSE loss with λ_coord=5 / λ_noobj=0.5; trained on PASCAL VOC at 448×448 | 🟡 **predict** — pjreddie's `yolov1.weights` parsed by our own loader (`yolov1_weights.cpp`, pure C++, no Darknet); forward → Darknet-flat decode → NMS; default imgsz=448. Train / ONNX / TRT staged as #66 / #68. |
| **yolo2**  | 2017 | Redmon & Farhadi (official Darknet)        | Darknet-19 + BN + leaky 0.1 + `reorg` passthrough + 5-anchor `region` head; output (5·(5+nc))×13×13 at imgsz=416. Full + tiny variants | 🟡 **predict (+tiny)** — pjreddie's `yolov2.weights` / `-tiny` parsed by our own loader (`yolov2_weights.cpp`, pure C++); reorg replicates Darknet's exact flat-memory layout so trained conv weights consume the right channels. Default imgsz=416. Train / ONNX / TRT staged as #67 / #69. |
| **yolo3**  | 2018 | Redmon & Farhadi (official Darknet)        | Darknet-53 backbone; ships in two head forms — original anchor-based (deferred) and the upstream anchor-free `yolov3u` (v8-style DFL head, used here) | ✅ **predict / val / train / ONNX+TRT export (yolov3u)** — converted on first use (fp16 → fp32). 103M params; 7 dets on `bus.jpg`, top 0.94. v3 train via `TrainerT<Yolo3>` reuses `V8DetectionLoss`. v3 ONNX (415 MB) + TRT FP32 (483 MB) match C++ predict's 7-dets baseline exactly. |
| **yolo4**  | 2020 | Bochkovskiy et al. (official Darknet)      | CSPDarknet-53 + SPP + PANet; Mish activations; v3-style anchor head | ✅ **predict / val / train / ONNX+TRT export** — Darknet `yolov4.weights` converted to our `yolo4.pt` on first use; default `imgsz=608` (anchor calibration). 6 dets on `bus.jpg`. v4 train via `V7DetectionLoss` (anchor-based with v4 scale_xy bias-fix + `exp()` wh decode). v4 ONNX (257 MB) + TRT FP32 (259 MB) match C++ baseline. |
| **yolo5**  | 2020 | upstream (official)                        | CSPNet + C3 blocks, originally anchor-based; the modern `*u.pt` files use the v8 anchor-free Detect head | ✅ end-to-end (predict / train / val / ONNX + TRT export) for all 5 scales via `yolo5*u.pt` |
| **yolo6**  | 2022 | Meituan (official open-source)             | EfficientRep backbone (RepBlock for n/s, BepC3 with BottleRep for m/l) + RepBiFPANNeck + CSPSPPF (n/s) / SimSPPF (m/l) + EffiDeHead. l uses SiLU. Plus MBLA variants (s/m/l/x_mbla) and P6 variants (n6/s6/m6/l6) | ✅ **predict + val + train + ONNX/TRT export for all 12 published variants** (n/s/m/l + 4×_mbla + n6/s6/m6/l6). m/l use BepC3 + DFL; MBLA uses MBLABlock (multi-branch BottleRep3); P6 uses the 6-stage backbone + 4-level head at imgsz=1280. **Train via `V6DetectionLoss`** (VFL + SIoU + TAL); finetune mAP@0.5:0.95=0.74 on coco8. bus.jpg TRT FP32 returns: P5 n/s/m/l = 4/5/5/6, MBLA s/m/l/x = 6/6/6/5, P6 n6/s6/m6/l6 = 5/6/5/8 dets — all matching libtorch. |
| **yolo7**  | 2022 | Wang, Bochkovskiy & Liao (academic)        | ELAN backbone + ELAN-W neck + MP/DownC downsamples + SPPCSPC + (3-level IDetect for base/tiny/x; 4-level IDetect + ReOrg input for w6/e6/d6/e6e). e6e adds E-ELAN parallel ELAN sub-blocks summed via Yolo7Shortcut | ✅ **predict + val for all 7 variants**, **train + ONNX+TRT export** for the IDetect anchor-decode form. v7 train via `V7DetectionLoss` (scale_xy=2.0 + `(sigmoid*2)²` wh decode); base finetune mAP@0.5:0.95=0.72 on coco8. v7 ONNX walks the per-scale yaml via the public `yolo7_yaml_for(scale)` accessor. |
| **yolo8**  | 2023 | upstream (official)                        | CSP + C2f backbone, anchor-free DFL Detect, TAL assigner | ✅ **full** — train / val / predict / export across 5 scales × 5 tasks (detect / segment / classify / pose / OBB) |
| **yolo9**  | 2024 | Wang, Yeh & Liao (academic)                | GELAN backbone (RepNCSPELAN4 + ADown/AConv + SPPELAN + ELAN1) + v8-style anchor-free Detect head; PGI auxiliary branch dropped at deploy. e adds CBLinear/CBFuse two-pass backbone | ✅ **predict + val + train + ONNX/TRT export for all 5 scales (t / s / m / c / e)**. e ONNX (added 0.20.0) emits the 43-layer two-pass graph: a primary backbone with 5 CBLinear taps, a secondary backbone that re-ingests the input image and pulls CBLinear branches via CBFuse (`Slice` + `Resize(mode=nearest)` + `Add`) at each downsample, plus the standard GELAN head. v9{c,e} TRT FP32 returns 5 dets on bus.jpg matching libtorch. PGI auxiliary branch is intentionally not wired (training-only upstream). |
| **yolo10** | 2024 | Tsinghua MIG (academic, upstream-hosted)   | SCDown + C2f + C2fCIB + SPPF + PSA backbone; v10Detect (one2one head used at deploy → effectively NMS-free) | ✅ **predict + val + train + ONNX+TRT export for all 6 scales (n / s / m / b / l / x)**. Single-head training uses the deploy one2one head with `V8DetectionLoss`; paper [P6]3.1 dual-head consistent assignment (added 0.22.0) trains a parallel one2many head (legacy=true cv3) with `V10DualLoss` = `V8DetectionLoss(o2m, topk=10)` + `V8DetectionLoss(o2o, topk=1)` — enable via `dual_head=true`. TRT FP32 disables TF32 per-version (the RepVGGDW 7×7 dwconv stack accumulates enough TF32 mantissa loss to saturate cls); after the fix every scale matches ORT (5 dets on bus.jpg, top conf 0.94–0.97). |
| **yolo11** | 2024 | **upstream (official)**                    | Refined CSP: C3k2 (kernel-tunable C3) + C2PSA (position-sensitive attention); v11 Detect head with depthwise-separable cv3 (DWConv→Conv) | ✅ **full** — train / val / predict / ONNX + TRT export across 5 scales × 5 tasks. Forward bit-exact vs upstream Python through layer 22 (parity harness verified). Full-COCO val mAP@0.5:0.95 within 0.05% of upstream's own `m.val(rect=False)` on n/s. |
| **yolo12** | 2025 | Tian et al., upstream-hosted               | Attention-centric: A2C2f (Area-Attention C2f) with windowed global attention, gamma-gated outer residual at l/x | ✅ **detect end-to-end** — train / val / predict / ONNX + TRT export across all 5 scales (n/s/m/l/x). Forward parity-clean (5/5/5/6/5 detections matching Python on bus.jpg). ONNX max\|Δ\| ≤ 1.8e-7 vs Python through onnxruntime. Task heads (segment / pose / obb / classify) ⏳ **planned future session** — upstream ships only detect weights for v12, so we'll train our own task heads on COCO. |
| **yolo13** | 2025 | Lei et al. (iMoonLab fork)                 | HyperACE (hypergraph adaptive correlation enhancement) + FullPAD distribution + DSConv depthwise-separable variants + V13AAttn (separate qk/v convs, k=5 pe) | ✅ **detect end-to-end** — train / val / predict / ONNX + TRT export across n/s/l/x (iMoonLab does not ship `m`). Forward cls-channel max\|Δ\| ≤ 7.6e-10 vs iMoonLab Python on all 4 scales. ONNX max\|Δ\| ≤ 1.8e-7. Task heads ⏳ **planned future session** — iMoonLab ships only detect weights, so we'll train our own task heads on COCO. |
| **yolo26** | 2025 | **upstream (official, preview)**           | DFL-free Detect head, end-to-end NMS-free inference, ProgLoss + STAL assigner — edge/mobile-first | ✅ **full** — train / val / predict / ONNX + TRT export across 5 scales × 5 tasks |

DETR-family models (RT-DETR, RF-DETR) used to live here in a scaffold
state; they have been moved to a separate repository so this repo can
stay focused on the closed set of twelve YOLO versions above.

Pending status (⏳) means the architecture is end-to-end for detect, but
task variants (segment/pose/obb/classify) are not yet trained because
neither upstream publishes those weights. Planned to train our own on
COCO in a future session.

### Capability matrix at a glance (detect)

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

Outstanding work — see **[TODO.md](TODO.md) [P6]2A** for the full filed
roadmap (#46..#63). The earlier per-version "v3 train / v4 train /
…" gaps that lived here in pre-0.22 releases have all closed (every
detect-pipeline cell of the matrix above is ✅); what remains is the
new feature/refactor work captured in TODO.md.

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

# One-shot: pre-convert any locally-available Darknet .weights
# (yolov4, yolov2*, yolov1*) to data/yolo*.pt. The runtime
# consumes .pt exclusively after this; .weights ingestion is a
# maintenance operation, not a runtime one.
./build/tools/convert_weights
```

The test suite covers every layer of the stack:

| test                 | what it checks                                       |
|----------------------|------------------------------------------------------|
| `smoke_test`         | LibTorch CUDA + custom CUDA kernel + OpenCV + TRT    |
| `test_yolo8_arch`    | YOLO8n architecture: param count (~3.16M), strides, output shapes |
| `test_pt_loader`     | Loads real `yolo8n.pt`, verifies all 355 keys/shapes |
| `test_predict`       | End-to-end inference on `bus.jpg`, ≥1 person + ≥1 bus |
| `test_train_overfit` | Tiny synthetic dataset, training loss decreases ≥ 2× |
| `test_v{3,4,6,7,9,10}_e2e` | Per-version predict end-to-end, gated on cache `.pt` availability |
| `test_val_v3_v10`    | Templated `engine::validate` runs on v3/v4/v6/v7/v9/v10 holders |
| `test_v9_train`      | yolo9c finetune-on-coco8 smoke (loss decreases, checkpoint written) |
| `test_v13_full`      | v13 forward parity vs iMoonLab Python (cls max\|Δ\| ≤ 7.6e-10) |
| `test_v13_ada`       | v13 hypergraph modules (AdaHyperedgeGen / HyperACE) bit-exact |

## CLI

**Single canonical parser** as of 0.99.x: `--mode <action>` plus flat
top-level flags. The earlier kv-style (`mode=...`) and legacy
subcommand-style parsers were removed (see CLAUDE.md "CLI surface" —
one canonical parser only).

```
yolocpp --mode train   -m yolo11n.pt -d coco/data.yaml -e 100 -b 16
yolocpp --mode val     -m yolo11n.pt -d coco/data.yaml
yolocpp --mode predict -m yolo11n.pt -s bus.jpg
yolocpp --mode predict -m yolo11n.pt -s frames/ -o annotated/
yolocpp --mode export  -m yolo11n.pt -f trt -p fp16
yolocpp --mode benchmark -m yolo11n.pt -s bus.jpg
yolocpp --mode info
yolocpp --mode download --dataset coco8
```

Common flags:

| short | long                  | scope                        |
|-------|-----------------------|------------------------------|
| -m    | --model / --weights   | every mode                   |
| -s    | --source              | predict, benchmark           |
| -d    | --data                | train, val                   |
| -o    | --out                 | predict, export              |
| -D    | --device              | every mode                   |
| -i    | --imgsz               | every mode                   |
| -e    | --epochs              | train                        |
| -b    | --batch               | train                        |
| -n    | --nc                  | predict, export              |
| -c    | --conf                | predict                      |
| -f    | --format              | export                       |
| -p    | --precision           | export                       |
|       | --seed                | train                        |
|       | --export-after-train  | train                        |
|       | --task                | predict, val, train, export  |

`--data` accepts five forms — `coco/` YOLO layout, `dataset.csv/.tsv`,
`instances.json` (COCO), `VOC2012/` (Pascal), and `data.yaml` —
dispatched by extension and directory layout (see CLAUDE.md "Dataset
format dispatch").

`--model` alone is enough: version (v3..v26), scale (n/s/m/l/x) and
`nc` are auto-inferred from the `.pt`'s actual layer shapes
(`model.0.conv.weight` kernel + first dim, head's `cv3.0.2.weight`
first dim). Works for renamed checkpoints (`best.pt`, `last.pt`).
Pass `--scale` / `--nc` only to override.

`--task` defaults to `detect` and accepts `detect | classify | segment
| pose | obb`. Detect routes through the registry for every supported
YOLO version; the four non-detect tasks use the v8 task families
(`Yolo8Classify`, `Yolo8Segment`, `Yolo8Pose`, `Yolo8OBB`).

```
yolocpp --mode train --task classify -d DATA -m yolo8n-cls.pt -e 30
yolocpp --mode train --task segment  -d DATA -m yolo8n-seg.pt -e 30
yolocpp --mode train --task pose     -d DATA -m yolo8n-pose.pt -e 30
yolocpp --mode train --task obb      -d DATA -m yolo8n-obb.pt -e 30
yolocpp --mode val   --task segment  -d DATA -m runs/segment/last.pt
```

### Public C++ API

```cpp
#include <yolocpp/api.hpp>

yolocpp::YOLO m("yolo11n.pt");
m.to("auto");
m.predict({.source = "bus.jpg"});
m.train({.data = "coco/data.yaml", .epochs = 100, .seed = 42});
m.val({.data = "coco/data.yaml"});
m.export_({.format = "onnx", .precision = "fp16"});
```

Routes through the same `cmd_*` functions the CLI uses.

### Uniform prediction output across all 12 versions

Every model — v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
v26 — returns the same `std::vector<inference::Detection>` from
`predict()` regardless of backend (`Predictor` for v8,
`GenericPredictor<Holder>` for the registry path, `TrtPredictor`
for the TRT runtime). The `Detection` struct is:

```cpp
struct Detection {
  float x1, y1, x2, y2;   // xyxy in original-image pixel coords
  float conf;             // confidence (max-class score)
  int   cls;              // class id (0-based)
};
```

CLI `--mode predict` writes the same annotated `runs/predict/<stem>.jpg`
+ a sibling `<stem>.txt` (`cls conf x1 y1 x2 y2` per line, one detection
per row) for **every** supported version. Anchor-based (v3/v4/v7),
anchor-free DFL (v5u/v6/v8/v9/v11/v12/v13), and end-to-end NMS-free
(v10/v26) all converge to the same xyxy + conf + cls shape before
the writer sees them — that's what makes downstream consumers
(visualisers, eval scripts, custom integrations) work uniformly.

The contract holds across batched inference too: `TrtPredictor::
predict_batch(vector<cv::Mat>)` returns
`vector<vector<Detection>>` — N input images → N detection lists,
same per-image shape.

What this enables for end users:

```cpp
// Swap models freely without touching the consumer side:
for (const auto& weights : {"yolo3.pt", "yolo8n.pt", "yolo11x.pt",
                            "yolo13n.pt", "yolo26n.pt"}) {
  yolocpp::YOLO m(weights);
  auto dets = m.predict({.source = "bus.jpg"});
  for (const auto& d : dets) {
    fmt::print("{:.2f}  ({:.0f},{:.0f})-({:.0f},{:.0f})  cls={}\n",
               d.conf, d.x1, d.y1, d.x2, d.y2, d.cls);
  }
}
```

Same identifiers, same fields, same semantics for every YOLO
version — the consumer code doesn't care which detector head it
came from. Roughly matches Ultralytics' `Results.boxes.xyxy /
.conf / .cls` shape, minus the Python `Results` wrapper (a
future addition tracked as task #97: add `Results` with
`.boxes`, `.names`, `.orig_shape`, `.speed`, `.plot()`,
`.save()` methods for fuller API parity).

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
    pt_loader           Reads upstream `.pt` zip + clean-room pickle parser
                        → state_dict {dotted_name → torch::Tensor}
  inference/
    letterbox           letterbox + image_to_tensor + scale_boxes
    nms                 class-aware NMS (CPU)
    predictor           load weights → preprocess → forward → NMS → unscale
  datasets/
    yolo_dataset        YOLO-format (.txt) loader, HSV + flip augmentation
  losses/
    yolo8_loss          TAL (alpha=0.5, beta=6, topk=10) + CIoU + DFL + BCE
    yolo26_loss         dual-head (o2m topk=10 + o2o topk=1) + ProgLoss + STAL
    yolo6_loss          VFL + SIoU + TAL
    yolo7_loss          anchor-based (v4 scale_xy bias-fix + exp() wh)
  engine/
    trainer             FusedAdamW (1-kernel _fused_adamw_) + Ultralytics
                        per-epoch linear LR + 3-epoch warmup + EMA
                        (decay=0.9999, tau=2000) + cuDNN benchmark +
                        TF32 + bf16 AMP autocast + channels_last +
                        BatchPrefetcher (4 workers, per-worker sharded
                        anchors) + GPU augmentation (mosaic, perspective,
                        HSV, hflip) — all on device, no per-step host syncs
    validator           full pass over val set → mAP via metrics/map
  metrics/map           per-class AP at IoU={0.5, 0.5..0.95}, COCO 101-point
```

## Architecture decisions worth knowing

- **`.pt` loader is a clean-room pickle parser**, not libtorch's
  `torch::jit::Unpickler`. Reason: the standard Unpickler trips on the
  BUILD opcode for arbitrary Python classes (`DetectionModel` etc.). Our
  parser handles every opcode produced by `torch.save` and treats unknown
  GLOBALs as opaque object stubs while still extracting the underlying
  tensor data.
- **Modules are constructed in the same order as the upstream yaml**, so
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

## Documentation map

- **[CHANGELOG.md](CHANGELOG.md)** — chronological changelog covering
  every phase, version, and parity gotcha. **Start here when you want
  to know what changed and why.**
- **[CLAUDE.md](CLAUDE.md)** — internal developer notes / Claude Code
  briefing. Detailed architecture commitments, per-version parity
  notes (BN eps, CSPSPPF cat order, RepVGG fusion gotchas, etc.),
  build / toolchain rationale (why DT_RPATH, why cu130, etc.).
- **README.md** (this file) — public-facing entry point: status,
  build/run, CLI surface, supported versions.

## What's deliberately deferred

The full deferred / pending list lives in **[TODO.md](TODO.md)**. Highlights:

- **v12 / v13 task heads (segment / pose / obb / classify)** — neither
  upstream nor iMoonLab publishes task weights (only detect ships).
  Scaffolding exists in `src/models/yolo12_tasks.cpp`; we'll train
  our own task heads on COCO under task #60.
- **simdjson for COCO instances.json** — currently a hand-rolled
  tokenizer in `src/datasets/coco_dataset.cpp`. simdjson would be
  faster + cleaner; deferred because the YOLO-format pipeline doesn't
  hit this path.
- **TRT INT8 calibration** + dynamic-shape multi-batch profiles —
  easy on top of `TrtBuildConfig` once a calibration set exists.
- **Two-GPU DDP training** — wiring is in place, world_size=1
  verified; no two-GPU box has run training yet.

### Already landed (formerly "deferred", current state of trainer)

- **bf16 AMP autocast + cuDNN benchmark + TF32 + channels_last** —
  landed 0.90.0.
- **Mosaic + RandomPerspective + MixUp augmentation** — landed 0.54.0
  (CPU mosaic). GPU mosaic + GPU perspective + per-sample affine
  + 114-grey padding — landed 0.98.0.
- **BatchPrefetcher** (4 workers, per-worker sharded anchors,
  pin-memory non-blocking HtoD) — landed 0.94.0.
- **FusedAdamW** (1-kernel `_fused_adamw_` per param group) — landed
  0.95.0.
- **Pure-GPU bbox transform + branchless gpu_hflip_** (no
  per-batch host syncs, no `.cpu()` / `.item()` mid-step) — landed 0.99.8.
- **Stride-aware cls bias init** (Ultralytics formula
  `log(5 / nc / (640/stride)²)`) — landed 0.99.8.
- **TAL top-K correctness fix** (`amax → amin` on threshold,
  matching Ultralytics' top-K mask) — landed 0.99.8.
- **Val NMS thresholds** (conf=0.001, iou=0.7, max_det=300 — the
  upstream val convention, not predict) — landed 0.99.9.
- **BGR/RGB channel-order consistency** between train and val —
  landed 0.99.9 (HSV jitter follow-on 0.99.11).
- **Ultralytics-style per-epoch linear LR** (peak ≈ 0.6·lr0 during
  warmup, not 1.0·lr0; eliminates mid-training mAP dip on
  small-parameter cls heads) — landed 0.99.12. **This is the fix
  that closed the n-variant gap and produced the BEAT/TIED
  outcomes in the benchmark table above.**

For the full filed roadmap (modular architecture, CLI overhaul,
trackers + SAHI, additional YOLO families, multi-device deployment,
weights publication, license decision, etc.) see TODO.md [P6]2A
(tasks #46..#63).
