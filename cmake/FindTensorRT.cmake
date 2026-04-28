# FindTensorRT.cmake
# Locates a vendored TensorRT installation under ${YOLOCPP_THIRD_PARTY}/tensorrt
# (extracted from NVIDIA .deb packages — see scripts/install_tensorrt.sh).
#
# Provides imported targets:
#   TensorRT::nvinfer
#   TensorRT::nvonnxparser
#   TensorRT::nvinfer_plugin
# Sets:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_VERSION_STRING

set(_trt_root "${YOLOCPP_THIRD_PARTY}/tensorrt")

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${_trt_root}/include" /usr/include/x86_64-linux-gnu /usr/include
)

find_library(TensorRT_nvinfer_LIBRARY
  NAMES nvinfer
  HINTS "${_trt_root}/lib" /usr/lib/x86_64-linux-gnu /usr/lib
)
find_library(TensorRT_nvonnxparser_LIBRARY
  NAMES nvonnxparser
  HINTS "${_trt_root}/lib" /usr/lib/x86_64-linux-gnu /usr/lib
)
find_library(TensorRT_nvinfer_plugin_LIBRARY
  NAMES nvinfer_plugin
  HINTS "${_trt_root}/lib" /usr/lib/x86_64-linux-gnu /usr/lib
)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(READ "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _v)
  # TRT 10.x defines TRT_*_ENTERPRISE and aliases NV_TENSORRT_* to them
  # at preprocessor time. Extract the literal numeric defines.
  string(REGEX MATCH "TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)" _ "${_v}")
  set(_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "TRT_MINOR_ENTERPRISE[ \t]+([0-9]+)" _ "${_v}")
  set(_minor "${CMAKE_MATCH_1}")
  string(REGEX MATCH "TRT_PATCH_ENTERPRISE[ \t]+([0-9]+)" _ "${_v}")
  set(_patch "${CMAKE_MATCH_1}")
  string(REGEX MATCH "TRT_BUILD_ENTERPRISE[ \t]+([0-9]+)" _ "${_v}")
  set(_build "${CMAKE_MATCH_1}")
  if(_major AND _minor AND _patch)
    set(TensorRT_VERSION_STRING "${_major}.${_minor}.${_patch}.${_build}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
  VERSION_VAR TensorRT_VERSION_STRING
)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")

  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
  endif()
  if(NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer
    )
  endif()
  if(NOT TARGET TensorRT::nvinfer_plugin)
    add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
      IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES TensorRT::nvinfer
    )
  endif()
endif()
