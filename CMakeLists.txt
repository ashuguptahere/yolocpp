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
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
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
  src/models/yolo3.cpp
  src/models/yolo4.cpp
  src/models/yolo5.cpp
  src/models/yolo6.cpp
  src/models/yolo7.cpp
  src/models/yolo8.cpp
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
  src/datasets/coco_dataset.cpp
  src/datasets/flat_dataset.cpp
  src/datasets/voc_dataset.cpp
  src/datasets/yolo_dataset.cpp
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
