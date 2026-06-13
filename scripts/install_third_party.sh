#!/usr/bin/env bash
# Idempotent installer for the C++ toolchain vendored under third_party/.
# Re-running this on a clean machine reproduces the layout used by CMake.
#
#   - LibTorch 2.11.0 + CUDA 13.0 (matches system /usr/local/cuda-13.0)
#   - TensorRT 10.14.1.48 + cuda13.0 (Blackwell sm_120 builder included)
#   - OpenCV 4.6.0 dev (Ubuntu 24.04 universe debs)
#   - CLI11 single-header
#
# Requires: curl, dpkg-deb, unzip, apt-get download. No sudo needed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"
cd "$TP"

# --- LibTorch -----------------------------------------------------------------
if [[ ! -d libtorch ]]; then
  echo "==> downloading LibTorch 2.11.0+cu130 (~1.7 GB)"
  curl -L --retry 5 --retry-delay 3 -C - -o libtorch.zip \
    "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip"
  unzip -q libtorch.zip
  rm libtorch.zip
fi

# --- TensorRT -----------------------------------------------------------------
if [[ ! -d tensorrt ]]; then
  echo "==> downloading TensorRT 10.14.1.48+cuda13.0 (~1.8 GB total, lib10 + headers)"
  mkdir -p tensorrt_debs tensorrt_root
  cd tensorrt_debs
  TRT_VER="10.14.1.48-1+cuda13.0"
  BASE="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64"
  for pkg in \
      libnvinfer10 \
      libnvinfer-headers-dev \
      libnvinfer-plugin10 libnvinfer-plugin-dev libnvinfer-headers-plugin-dev \
      libnvonnxparsers10 libnvonnxparsers-dev; do
    curl -sL --retry 5 -o "${pkg}.deb" "${BASE}/${pkg}_${TRT_VER}_amd64.deb"
  done
  cd "$TP"
  for f in tensorrt_debs/*.deb; do dpkg-deb -x "$f" tensorrt_root/; done
  ln -sf libnvinfer.so.10 tensorrt_root/usr/lib/x86_64-linux-gnu/libnvinfer.so
  mkdir -p tensorrt
  ln -sfn "$TP/tensorrt_root/usr/include/x86_64-linux-gnu" tensorrt/include
  ln -sfn "$TP/tensorrt_root/usr/lib/x86_64-linux-gnu"     tensorrt/lib
  rm -rf tensorrt_debs
fi

# --- OpenCV -------------------------------------------------------------------
if [[ ! -d opencv ]]; then
  echo "==> downloading OpenCV 4.6.0 dev + runtime debs (~25 MB)"
  mkdir -p opencv_debs opencv_root
  cd opencv_debs
  for pkg in \
      libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev \
      libopencv-videoio-dev libopencv-dnn-dev libopencv-flann-dev \
      libopencv-features2d-dev libopencv-calib3d-dev libopencv-highgui-dev \
      libopencv-ml-dev libopencv-objdetect-dev libopencv-photo-dev \
      libopencv-stitching-dev libopencv-video-dev libopencv-contrib-dev \
      libopencv-superres-dev libopencv-videostab-dev libopencv-shape-dev \
      libopencv-viz-dev libopencv-dev \
      libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
      libopencv-videoio406t64 libopencv-dnn406t64 libopencv-flann406t64 \
      libopencv-features2d406t64 libopencv-calib3d406t64 libopencv-highgui406t64 \
      libopencv-ml406t64 libopencv-objdetect406t64 libopencv-photo406t64 \
      libopencv-stitching406t64 libopencv-video406t64 libopencv-contrib406t64 \
      libopencv-superres406t64 libopencv-videostab406t64 libopencv-shape406t64 \
      libopencv-viz406t64; do
    apt-get download "$pkg" 2>/dev/null || echo "  (skip $pkg)"
  done
  cd "$TP"
  for f in opencv_debs/*.deb; do dpkg-deb -x "$f" opencv_root/; done
  # If a runtime .so.4.6.0 isn't in our extract (because that runtime deb
  # was missing), symlink from system /usr/lib/x86_64-linux-gnu when present.
  cd opencv_root/usr/lib/x86_64-linux-gnu
  for sys in /usr/lib/x86_64-linux-gnu/libopencv_*.so.4.6.0; do
    [[ -e "$sys" && ! -e "$(basename "$sys")" ]] && ln -sf "$sys" "$(basename "$sys")"
  done
  cd "$TP"
  mkdir -p opencv
  ln -sfn "$TP/opencv_root/usr/include" opencv/include
  ln -sfn "$TP/opencv_root/usr/lib"     opencv/lib
  rm -rf opencv_debs
fi

# --- single-header libs (CLI11, rapidyaml, clay, cpp-httplib) -----------------
# No longer fetched here: the build pulls + sha-verifies them at configure time
# via cmake/dependencies.cmake (the single source of truth for every pin), or
# reuses third_party/<lib>/ when present (CLI11 + rapidyaml are committed for
# offline builds). Nothing to do in this script for them.

echo "==> third-party install OK"
echo "    $TP/libtorch  -> LibTorch $(cat "$TP/libtorch/build-version" 2>/dev/null || echo unknown)"
echo "    $TP/tensorrt  -> $(grep -oP 'TRT_MAJOR_ENTERPRISE\s+\K[0-9]+' "$TP/tensorrt/include/NvInferVersion.h").$(grep -oP 'TRT_MINOR_ENTERPRISE\s+\K[0-9]+' "$TP/tensorrt/include/NvInferVersion.h").$(grep -oP 'TRT_PATCH_ENTERPRISE\s+\K[0-9]+' "$TP/tensorrt/include/NvInferVersion.h")"
echo "    $TP/opencv    -> 4.6.0"
echo "    $TP/CLI11     -> 2.6.2"
