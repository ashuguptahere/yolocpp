# ───────────────────────────────────────────────────────────────────────────
# Centralized dependency manifest — single source of truth for every pin.
#
# Two acquisition paths, both CMake-driven:
#   • Portable libs (CLI11, rapidyaml, clay, cpp-httplib) → pinned single
#     headers pulled by `file(DOWNLOAD … EXPECTED_HASH)` at configure time, or
#     reused from third_party/<dir> if the install script already placed them
#     (offline / existing trees). Exposed as INTERFACE targets.
#   • LibTorch → the pinned prebuilt cu130 zip, pulled by FetchContent (reused
#     from third_party/libtorch when present so existing trees don't re-download
#     1.7 GB).
#
# The GPU `.deb` stack (TensorRT, OpenCV) cannot be acquired by CMake — `.deb`
# isn't a FetchContent archive — so their *versions* are pinned here as
# variables and `scripts/install_third_party.sh` does the `dpkg-deb -x`
# extraction; CUDA is the system toolkit. DEPS.md mirrors this table for humans.
# ───────────────────────────────────────────────────────────────────────────

include(FetchContent)

# ---- pins (the one place to bump a version) -------------------------------
set(YOLOCPP_LIBTORCH_VERSION "2.11.0+cu130")
set(YOLOCPP_LIBTORCH_URL "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip")
set(YOLOCPP_TENSORRT_VERSION "10.14.1.48")   # extracted from .deb by install script
set(YOLOCPP_OPENCV_VERSION   "4.6.0")         # extracted from .deb by install script
set(YOLOCPP_CUDA_VERSION     "13.0")          # system toolkit

# Portable single-header deps: target | third_party-dir | include-filename | url | sha256
set(YOLOCPP_CLI11_VERSION    "2.6.2")
set(YOLOCPP_RYML_VERSION     "0.11.1")
set(YOLOCPP_CLAY_VERSION     "0.14")
set(YOLOCPP_HTTPLIB_VERSION  "0.46.1")

# ---- header-only dep helper ------------------------------------------------
# Reuse third_party/<dir>/<outfile> if present, else pull + sha-verify into
# build/_deps/<tgt>/<outfile>. Always creates INTERFACE target <tgt>.
function(yolocpp_header_dep tgt dir outfile url sha)
  set(_tp "${YOLOCPP_THIRD_PARTY}/${dir}")
  if(EXISTS "${_tp}/${outfile}")
    set(_inc "${_tp}")
    message(STATUS "dep ${tgt}: vendored ${_tp}/${outfile}")
  else()
    set(_inc "${CMAKE_BINARY_DIR}/_deps/${tgt}")
    if(NOT EXISTS "${_inc}/${outfile}")
      message(STATUS "dep ${tgt}: fetching ${url}")
      file(DOWNLOAD "${url}" "${_inc}/${outfile}"
           EXPECTED_HASH SHA256=${sha} TLS_VERIFY ON SHOW_PROGRESS STATUS _st)
      list(GET _st 0 _code)
      if(NOT _code EQUAL 0)
        file(REMOVE "${_inc}/${outfile}")
        message(FATAL_ERROR "dep ${tgt}: download failed (${_st}) from ${url}")
      endif()
    endif()
  endif()
  add_library(${tgt} INTERFACE)
  target_include_directories(${tgt} INTERFACE "${_inc}")
endfunction()

yolocpp_header_dep(CLI11 CLI11 CLI11.hpp
  "https://github.com/CLIUtils/CLI11/releases/download/v${YOLOCPP_CLI11_VERSION}/CLI11.hpp"
  "227a16fe5f9f8ada80c3c409492475536f597e7bd83a6c26eacc3c8c149a9295")

yolocpp_header_dep(rapidyaml rapidyaml ryml_all.hpp
  "https://github.com/biojppm/rapidyaml/releases/download/v${YOLOCPP_RYML_VERSION}/rapidyaml-${YOLOCPP_RYML_VERSION}.hpp"
  "6c068513694cba1885da23c96d6f8e1bf19bf090ee186efe9565f3db71903d63")

yolocpp_header_dep(clay clay clay.h
  "https://raw.githubusercontent.com/nicbarker/clay/v${YOLOCPP_CLAY_VERSION}/clay.h"
  "c97241cc423af3fa11267978adce9cbb46274a2ad0709a5d4b2b1092dc27599d")

yolocpp_header_dep(httplib httplib httplib.h
  "https://raw.githubusercontent.com/yhirose/cpp-httplib/v${YOLOCPP_HTTPLIB_VERSION}/httplib.h"
  "6ea64fcedb27668134c442b087ef854a487edc5e1328d8fbc3e919b2f55b5663")

# ---- LibTorch (FetchContent the pinned cu130 zip; reuse third_party first) -
if(NOT DEFINED Torch_DIR AND EXISTS "${YOLOCPP_THIRD_PARTY}/libtorch/share/cmake/Torch")
  set(Torch_DIR "${YOLOCPP_THIRD_PARTY}/libtorch/share/cmake/Torch" CACHE PATH "")
  message(STATUS "dep libtorch: vendored ${YOLOCPP_THIRD_PARTY}/libtorch")
elseif(NOT DEFINED Torch_DIR)
  message(STATUS "dep libtorch: fetching ${YOLOCPP_LIBTORCH_URL}")
  FetchContent_Declare(libtorch URL "${YOLOCPP_LIBTORCH_URL}")
  FetchContent_MakeAvailable(libtorch)
  set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch" CACHE PATH "")
endif()
find_package(Torch REQUIRED)
message(STATUS "LibTorch ${YOLOCPP_LIBTORCH_VERSION}: ${TORCH_LIBRARIES}")

# ---- CUDA toolkit ----------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} @ ${CUDAToolkit_INCLUDE_DIRS}")

# ---- TensorRT (cmake/FindTensorRT.cmake; .deb-extracted by install script) -
find_package(TensorRT REQUIRED)
message(STATUS "TensorRT ${TensorRT_VERSION_STRING} (pin ${YOLOCPP_TENSORRT_VERSION}) @ ${TensorRT_INCLUDE_DIRS}")

# ---- OpenCV (.deb-extracted by install script) -----------------------------
foreach(_p
    "${YOLOCPP_THIRD_PARTY}/opencv/lib/x86_64-linux-gnu/cmake/opencv4"
    "${YOLOCPP_THIRD_PARTY}/opencv/lib/cmake/opencv4")
  if(NOT DEFINED OpenCV_DIR AND EXISTS "${_p}/OpenCVConfig.cmake")
    set(OpenCV_DIR "${_p}" CACHE PATH "" FORCE)
  endif()
endforeach()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
message(STATUS "OpenCV ${OpenCV_VERSION} (pin ${YOLOCPP_OPENCV_VERSION}) @ ${OpenCV_INCLUDE_DIRS}")
