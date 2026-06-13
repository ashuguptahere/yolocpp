# yolocpp dependency manifest

The closed set of third-party dependencies, with pinned versions and the
reason each one is in the build. Anything not in this table should be
challenged before it lands. Maintained as part of #48 (deps audit).

## Build-time

| dep         | version           | why                                         | license      | source                                                            | size on disk |
|-------------|-------------------|---------------------------------------------|--------------|-------------------------------------------------------------------|--------------|
| libtorch    | 2.11.0+cu130      | training + eval; tensor + autograd          | BSD-3-Clause | https://download.pytorch.org/libtorch/cu130/                      | ~3.0 GB      |
| TensorRT    | 10.14.1.48+cuda13 | ONNX → engine builder + runtime predictor   | NVIDIA EULA  | NVIDIA developer portal (vendored .deb, extracted by script)      | ~2.5 GB      |
| CUDA        | 13.0.88           | runtime kernels (libcudart, libnvrtc)       | NVIDIA EULA  | system `/usr/local/cuda` (`apt install cuda-toolkit-13-0`)        | system       |
| OpenCV      | 4.6.0             | image / video I/O, letterbox, drawing       | Apache-2.0   | Ubuntu 24.04 universe debs (libopencv-* / libopencv-*-dev)        | ~124 MB      |
| NCCL        | 2.23.4            | DDP all-reduce (multi-GPU training)         | BSD-3-Clause | NVIDIA developer portal (vendored deb, extracted by script)       | ~52 KB (headers only — runtime via libnccl) |
| rapidyaml   | 0.11.1            | parse `data.yaml` (path/train/val/names)    | MIT          | biojppm/rapidyaml single-header amalgamation                      | ~1.7 MB      |
| CLI11       | 2.6.2             | flag-style CLI parser                       | BSD-3-Clause | CLIUtils/CLI11 single-header release (pinned tag + sha256-verified)| ~468 KB      |
| clay        | 0.14              | UI layout engine for the web console (`yolocpp_web`); server-side Clay→HTML | zlib | nicbarker/clay single-header (`clay.h`)               | ~300 KB      |
| cpp-httplib | 0.46.1            | HTTP/1.1 server backing the web console     | MIT          | yhirose/cpp-httplib single-header (`httplib.h`)                   | ~690 KB      |

Total: **~5.5 GB on disk**, ~99 % of which is libtorch + TensorRT.

**Pins live in [`cmake/dependencies.cmake`](../cmake/dependencies.cmake)** — the
single source of truth. CMake pulls + sha256-verifies the portable single
headers (CLI11, rapidyaml, clay, cpp-httplib) at configure time, or reuses
`third_party/<lib>/` when present (CLI11 + rapidyaml are committed for offline
builds). LibTorch is pulled by FetchContent from the pinned cu130 zip (reused
from `third_party/libtorch` when present). The `.deb` GPU stack (TensorRT,
OpenCV) and the system CUDA toolkit can't be CMake-pulled; their versions are
pinned in that module and `scripts/install_third_party.sh` does the extraction.
This table mirrors those pins for humans.

`clay` + `cpp-httplib` back the optional `yolocpp_web` target (build with
`-DYOLOCPP_BUILD_WEB=ON`, default on). Both are header-only, permissively
licensed (zlib / MIT), and pulled in only by `src/web/`. They add no runtime
`.so`. The web backend reuses the existing `yolocpp::YOLO` API for all GPU
work — the browser never links LibTorch.

## Runtime

| dep            | role                                                                         |
|----------------|------------------------------------------------------------------------------|
| libtorch.so    | tensor + module runtime                                                      |
| libnvinfer.so  | TRT engine deserialise / execute                                             |
| libcudart.so   | CUDA runtime                                                                 |
| libopencv_*.so | image / video I/O                                                            |
| libnccl.so     | DDP all-reduce (only when training with `--device cuda:0,1,...`)             |

**No Python at runtime.** Anything mentioning `python`, `torch.jit`, or
`onnxruntime` in this codebase is either dev-only tooling outside the
build (`/tmp/yolocpp_parity/`), or a comment about upstream behaviour
we mirror in C++.

## Why these and not others

- **No Boost** — every helper we need (filesystem, regex, optional, …)
  is in C++20 standard or libtorch already.
- **No protobuf** — we hand-write the ONNX wire format
  (`src/serialization/onnx_export.cpp`) instead of linking 30+ MB of
  libprotobuf to emit a single graph schema.
- **No GTest / Catch2** — tests are plain `int main()` files with an
  `EXPECT(cond, msg)` macro. Adding a test framework would dwarf the
  test code itself.
- **No fmt / spdlog** — `std::format` (C++20) and `std::cerr` cover
  every logging callsite.
- **No json library** — `data.yaml` is YAML and rapidyaml handles it.
  COCO JSON support (#54B) landed via a ~180-line hand-rolled streaming
  tokenizer (`src/datasets/coco_dataset.cpp` `JsonLexer`), which passed
  the ≤200-line-vs-new-dep test below; nlohmann/json, rapidjson and
  simdjson were evaluated and not added (a single fixed schema doesn't
  justify a parser dependency, and parsing is one-shot at dataset load,
  not a hot path).
- **No ONNX Runtime** — we go straight to TensorRT for deployment
  inference. Adding ORT would let us run on platforms without TRT
  (CPU-only, Apple Silicon) but that's #58 (multi-device deploy)
  territory.

## Adding a new dependency — checklist

1. Is it actually needed, or can the equivalent be written in ≤200
   lines of C++ in our tree? (rapidyaml passed this test; protobuf
   didn't.)
2. Does it have a stable single-header / header-only form? Vendor
   it under `third_party/<name>/<version>.hpp` so build is hermetic.
3. Otherwise, add an extraction step to `scripts/install_third_party.sh`
   pinning a specific tag / commit hash. No "latest".
4. Add a row to the table above with version + license + purpose +
   size. PRs that touch `CMakeLists.txt` to introduce a new
   `find_package` / `target_link_libraries` without updating this
   manifest fail the audit script (`scripts/audit_deps.sh`).
5. License must be Apache-2.0 / MIT / BSD-* compatible. GPL adds a
   contagion problem we explicitly avoid (see #50 reasoning).

## Build-time-only — cleanup

The `third_party/{tensorrt_root,opencv_root}` directories are full
`.deb` extractions. The `tensorrt/` and `opencv/` peers are
symlink-only adapters that the CMake `find_package` glue needs to
discover them at configure time. Removing files inside the `_root`
trees breaks the symlinks. To shrink them, update
`scripts/install_third_party.sh` to extract only the components we
link against (`libnvinfer*`, `nvinfer.h`, `NvOnnxParser.h`, etc.) —
filed as #48C if the disk pressure warrants it.
