# `runs/compare/` — paired runs across yolocpp and Ultralytics

Each sub-directory under `runs/compare/<mode>/<N>/` holds the results
of **one comparison run** for one variant: same model, same dataset,
same seed — different stack. Mirror of Ultralytics' `runs/` layout
but doubled across both codebases.

```
runs/compare/
├── train/
│   ├── 1/                              # corresponding run number
│   │   ├── manifest.yaml               # variant, epochs, batch, seed, …
│   │   ├── yolocpp/                    # yolocpp output
│   │   │   ├── args.yaml
│   │   │   ├── results.csv
│   │   │   ├── best.pt
│   │   │   ├── last.pt
│   │   │   ├── BoxPR_curve.png
│   │   │   ├── time.log                # /usr/bin/time -v output
│   │   │   └── gpu.csv                 # nvidia-smi polling (100 ms)
│   │   └── ultralytics/                # Ultralytics output
│   │       ├── args.yaml
│   │       ├── results.csv
│   │       ├── weights/best.pt
│   │       ├── weights/last.pt
│   │       ├── time.log
│   │       └── gpu.csv
│   ├── 2/ …
│   └── …
├── predict/                            # corresponding predict runs
├── val/                                # corresponding val runs
└── export/                             # corresponding export runs
```

## Manifest format

Each numbered sub-dir's `manifest.yaml` is the **single source of
truth** for what was compared:

```yaml
variant:   yolo11n
mode:      train
epochs:    5
batch:     16              # if different per side, list both: {yolocpp: 16, ultralytics: 16}
imgsz:     640
seed:      42
dataset:   screen-dataset
created:   2026-06-03
yolocpp_version: 0.99.45
ultralytics_version: 8.4.60
common_args:
  optimizer: auto
  workers:   0             # Blackwell deadlock workaround on Ultra side
note: ""
```

## Discovery / indexing

To find a comparison run by variant or by date, the top-level
`runs/compare/index.csv` is the lookup table:

```
run_id, mode, variant, epochs, batch, imgsz, seed, created
train/1, train, yolo11n, 5, 16, 640, 42, 2026-06-03
train/2, train, yolo11s, 5, 16, 640, 42, 2026-06-03
…
```

## Why a parallel structure (not the existing `runs/train/`)

The existing `runs/train/` is yolocpp-only and has 458 directories
spanning months of work. Restructuring it would break the documented
convention in CLAUDE.md (`Output convention (current) … runs/train/`).
Instead `runs/compare/` is the **new** directory for paired runs —
forward-only, no historical data restructured.

For a NEW comparison: write to `runs/compare/<mode>/<N>/<stack>/`
and update `runs/compare/index.csv`. For existing yolocpp-only
results: they stay in `runs/train/`, `runs/predict/`, etc. as before.
