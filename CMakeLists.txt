cmake_minimum_required(VERSION 3.24)

# Single source of truth for the project version: the top-level VERSION
# file. CMake reads it here, feeds it to project(... VERSION ...), and
# CMake then exports PROJECT_VERSION_{MAJOR,MINOR,PATCH} into
# yolocpp/config.hpp via cmake/yolocpp_config.hpp.in. Every other
# consumer (the CLI's `--version` flag, `yolocpp info`, docs, release
# tooling) reads YOLOCPP_VERSION_STRING from that header — never a
# duplicate literal. To bump the version, edit ./VERSION only.
file(READ "${CMAKE_SOURCE_DIR}/VERSION" YOLOCPP_VERSION_RAW)
string(STRIP "${YOLOCPP_VERSION_RAW}" YOLOCPP_VERSION)
if(NOT YOLOCPP_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION file must contain a single MAJOR.MINOR.PATCH line, got: '${YOLOCPP_VERSION}'")
endif()

project(yolocpp
  VERSION ${YOLOCPP_VERSION}
  DESCRIPTION "Pure C++ computer-vision suite (LibTorch + TensorRT)"
  LANGUAGES C CXX CUDA
)
# Pre-1.0 versioning policy lives in CHANGELOG.md. Bump the third digit
# (PATCH) for additive changes; bump the second (MINOR) when a new YOLO
# version, task, or pipeline lands. Stay on 0.x.y until the user
# declares the project 1.0-ready. To bump: edit ./VERSION (one line),
# then add a CHANGELOG entry.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable AVX2/FMA + AES (x86_64): inner-loop SIMD in NMS + future
# hot paths uses intrinsics that require -mavx2 -mfma. Modern Intel
# (Haswell+) and AMD (Excavator+) have AVX2; we don't target pre-2013
# hardware. -ffast-math is intentionally NOT set — keeps FP semantics
# strict (NaN handling for losses, exact mAP math).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-mavx2 -mfma)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Dev-loop accelerators — opportunistic. Each one is a no-op if the tool
# isn't installed, so the build still works with stock GCC + GNU ld;
# but if the user `apt install`s these, the next reconfigure picks them
# up automatically. See CLAUDE.md "Build + run speed".
#
#   - ccache:  caches compilation outputs by content hash — biggest
#              dev-loop win on incremental rebuilds. ~5-10× on warm
#              edits.
#   - mold:    massively-parallel linker, the only thing fast enough
#              to make the libtorch link enjoyable. Falls back to lld
#              if mold is missing; falls back to BFD if neither.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  # Inject CCACHE_SLOPPINESS via a `cmake -E env` launcher wrapper so ccache
  # can cache precompiled-header-using TUs. Without pch_defines/time_macros,
  # ccache silently refuses to cache PCH'd compiles, which would regress the
  # warm rebuild path the moment PCH is enabled (see YOLOCPP_ENABLE_PCH).
  set(CMAKE_CXX_COMPILER_LAUNCHER
      "${CMAKE_COMMAND}" -E env "CCACHE_SLOPPINESS=pch_defines,time_macros"
      "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "ccache: ${CCACHE_PROGRAM} (compiler launcher wired)")
endif()
find_program(MOLD_PROGRAM mold)
find_program(LLD_PROGRAM  lld)
if(MOLD_PROGRAM)
  add_link_options(-fuse-ld=mold)
  message(STATUS "linker: mold (${MOLD_PROGRAM})")
elseif(LLD_PROGRAM)
  add_link_options(-fuse-ld=lld)
  message(STATUS "linker: lld (${LLD_PROGRAM})")
endif()

# RTX 5090 = Blackwell sm_120. We compile virtual PTX for forward-compat too.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 90 120)
endif()
# Tell PyTorch's Caffe2 cmake which arches to target — it ignores
# CMAKE_CUDA_ARCHITECTURES and uses its own list.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST "8.9;9.0;12.0" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---- third-party prefix (header-only + vendored binaries) -----------------
set(YOLOCPP_THIRD_PARTY "${CMAKE_SOURCE_DIR}/third_party")

# Shared rpath used by every executable target.
set(_yolocpp_rpath
  "${YOLOCPP_THIRD_PARTY}/libtorch/lib"
  "${YOLOCPP_THIRD_PARTY}/tensorrt/lib"
  "${YOLOCPP_THIRD_PARTY}/opencv/lib/x86_64-linux-gnu"
)

# ---- LibTorch -------------------------------------------------------------
if(NOT DEFINED Torch_DIR AND EXISTS "${YOLOCPP_THIRD_PARTY}/libtorch/share/cmake/Torch")
  set(Torch_DIR "${YOLOCPP_THIRD_PARTY}/libtorch/share/cmake/Torch" CACHE PATH "")
endif()
find_package(Torch REQUIRED)
message(STATUS "LibTorch: ${TORCH_LIBRARIES}")

# ---- CUDA toolkit (we already enabled language CUDA above) ----------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} @ ${CUDAToolkit_INCLUDE_DIRS}")

# ---- TensorRT (optional at config time, hard-required at link time) -------
find_package(TensorRT REQUIRED)
message(STATUS "TensorRT: ${TensorRT_VERSION_STRING} @ ${TensorRT_INCLUDE_DIRS}")

# ---- OpenCV ---------------------------------------------------------------
# Vendored layout (extracted from Ubuntu .debs):
#   third_party/opencv/lib -> .../usr/lib (symlink)
#   third_party/opencv/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake
foreach(_p
    "${YOLOCPP_THIRD_PARTY}/opencv/lib/x86_64-linux-gnu/cmake/opencv4"
    "${YOLOCPP_THIRD_PARTY}/opencv/lib/cmake/opencv4")
  if(NOT DEFINED OpenCV_DIR AND EXISTS "${_p}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${_p}" CACHE PATH "" FORCE)
  endif()
endforeach()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
message(STATUS "OpenCV: ${OpenCV_VERSION} @ ${OpenCV_INCLUDE_DIRS}")

# ---- CLI11 (header-only vendored) -----------------------------------------
add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE "${YOLOCPP_THIRD_PARTY}/CLI11")

# ---- rapidyaml (header-only vendored) -------------------------------------
add_library(rapidyaml INTERFACE)
target_include_directories(rapidyaml INTERFACE "${YOLOCPP_THIRD_PARTY}/rapidyaml")

# ---- internal config header -----------------------------------------------
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/yolocpp_config.hpp.in"
  "${CMAKE_BINARY_DIR}/generated/yolocpp/config.hpp"
  @ONLY
)

# ---- core library ---------------------------------------------------------
add_library(yolocpp_core
  src/core/version.cpp
  src/core/device.cpp
  src/core/profile.cpp
  src/models/yolo3.cpp
  src/models/yolo4.cpp
  src/models/yolo5.cpp
  src/models/yolo6.cpp
  src/models/yolo7.cpp
  src/models/yolo8.cpp
  src/models/yolo1.cpp
  src/models/yolo2.cpp
  src/models/yolo8_classify.cpp
  src/models/yolo8_tasks.cpp
  src/models/yolo9.cpp
  src/models/yolo10.cpp
  src/models/yolo11.cpp
  src/models/yolo11_tasks.cpp
  src/models/yolo12.cpp
  src/models/yolo12_tasks.cpp
  src/models/yolo13.cpp
  src/models/yolo26.cpp
  src/models/yolo26_tasks.cpp
  src/inference/task_predictors.cpp
  src/tasks/classify_train.cpp
  src/tasks/segment_train.cpp
  src/tasks/pose_obb_train.cpp
  src/serialization/pt_loader.cpp
  src/serialization/pt_save.cpp
  src/serialization/darknet_weights.cpp
  src/serialization/yolov1_weights.cpp
  src/serialization/yolov2_weights.cpp
  src/serialization/yolov6_weights.cpp
  src/serialization/yolov7_weights.cpp
  src/serialization/yolov9_weights.cpp
  src/serialization/yolov10_weights.cpp
  src/serialization/yolov3_weights.cpp
  src/serialization/onnx_export.cpp
  src/serialization/trt_export.cpp
  src/inference/trt_predictor.cpp
  src/inference/letterbox.cpp
  src/inference/nms.cpp
  src/inference/predictor.cpp
  src/inference/results.cpp
  src/datasets/augment.cpp
  src/datasets/coco_dataset.cpp
  src/datasets/flat_dataset.cpp
  src/datasets/voc_dataset.cpp
  src/datasets/yolo_dataset.cpp
  src/losses/yolo1_loss.cpp
  src/losses/yolo2_loss.cpp
  src/losses/yolo8_loss.cpp
  src/losses/yolo26_loss.cpp
  src/losses/yolo6_loss.cpp
  src/losses/yolo7_loss.cpp
  src/losses/yolo10_loss.cpp
  src/cli/api.cpp
  src/cli/commands.cpp
  src/cli/data_yaml.cpp
  src/cli/model_info.cpp
  src/cli/resolve.cpp
  src/registry/version_registry.cpp
  src/engine/benchmark.cpp
  src/engine/ddp.cpp
  src/engine/trainer.cpp
  src/engine/validator.cpp
  src/metrics/map.cpp
)
target_include_directories(yolocpp_core
  PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/generated"
    ${TORCH_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${YOLOCPP_THIRD_PARTY}/nccl/include"
)
target_link_libraries(yolocpp_core
  PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cudart
    TensorRT::nvinfer
    TensorRT::nvonnxparser
    TensorRT::nvinfer_plugin
    ${OpenCV_LIBS}
  PRIVATE
    rapidyaml
)
target_compile_features(yolocpp_core PUBLIC cxx_std_20)

# ---- Precompiled headers --------------------------------------------------
# torch/torch.h is a multi-second-per-TU parse; PCH'ing it (plus the OpenCV
# umbrella headers used across the loaders/predictors) once and reusing it
# across the core TUs is the biggest cold-build / ccache-miss win after
# ccache itself. Opt-in (default ON) so a ccache-warm dev loop can disable
# it — PCH'd TUs are only ccache-cached when CCACHE_SLOPPINESS includes
# `pch_defines,time_macros`, so on a warm tree without that env the plain
# `-DYOLOCPP_ENABLE_PCH=OFF` path keeps ccache hits. No `.cu` sources live in
# yolocpp_core, so PCH is safe here.
option(YOLOCPP_ENABLE_PCH "Precompile heavy headers (torch/opencv)" ON)
if(YOLOCPP_ENABLE_PCH)
  target_precompile_headers(yolocpp_core PRIVATE
    <torch/torch.h>
    <opencv2/core.hpp>
    <opencv2/imgproc.hpp>
    <opencv2/imgcodecs.hpp>
  )
  message(STATUS "PCH: enabled (torch + opencv core/imgproc/imgcodecs)")
endif()

# ---- CLI executable -------------------------------------------------------
add_executable(yolocpp src/cli/main.cpp)
target_link_libraries(yolocpp PRIVATE yolocpp_core CLI11)

# ---- tests ----------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---- examples ------------------------------------------------------------
# Demonstrate the public C++ API (`#include <yolocpp/api.hpp>`). Toggleable
# via -DYOLOCPP_BUILD_EXAMPLES=OFF. See examples/README.md.
add_subdirectory(examples)

# ---- tools ---------------------------------------------------------------
# One-shot maintenance binaries (e.g. `convert_weights` — pre-converts
# Darknet `.weights` to `.pt`). Toggleable via -DYOLOCPP_BUILD_TOOLS=OFF.
add_subdirectory(tools)

# ---- runtime rpath so vendored .so's resolve at exec time -----------------
# Use DT_RPATH (transitive) instead of DT_RUNPATH so that dlopen() calls made
# by libnvinfer.so for its sm_* resource libraries also search the rpath.
# Without this, libnvinfer fails with "Unable to load library:
# libnvinfer_builder_resource_sm120.so.10.14.1" at engine build time.
target_link_options(yolocpp PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(yolocpp PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)
