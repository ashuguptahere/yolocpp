# Example programs that demonstrate the public C++ API
# (`#include <yolocpp/api.hpp>`).
#
# Each one is a self-contained executable showing one workflow.
# They link against `yolocpp_core` (the same library `yolocpp` does)
# and inherit the rpath / disable-new-dtags flags so libnvinfer's
# dlopen-by-rpath behaviour keeps working.
#
# Toggle `-DYOLOCPP_BUILD_EXAMPLES=OFF` to skip building these
# (default ON because they're cheap and they catch API regressions).

option(YOLOCPP_BUILD_EXAMPLES "Build the public-API example programs" ON)
if(NOT YOLOCPP_BUILD_EXAMPLES)
  return()
endif()

function(yolocpp_example name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE yolocpp_core)
  target_link_options(${name} PRIVATE "LINKER:--disable-new-dtags")
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH   "${_yolocpp_rpath}"
    INSTALL_RPATH "${_yolocpp_rpath}"
  )
endfunction()

yolocpp_example(predict_image)
yolocpp_example(predict_directory)
yolocpp_example(predict_video)
yolocpp_example(train_finetune)
yolocpp_example(export_to_onnx)
yolocpp_example(end_to_end)
yolocpp_example(benchmark_model)
