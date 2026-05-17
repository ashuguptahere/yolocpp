# Small one-shot maintenance tools that aren't part of the public API.
# They link against `yolocpp_core` and inherit the same rpath /
# disable-new-dtags treatment as everything else.

option(YOLOCPP_BUILD_TOOLS "Build the maintenance tool binaries" ON)
if(NOT YOLOCPP_BUILD_TOOLS)
  return()
endif()

function(yolocpp_tool name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE yolocpp_core)
  target_link_options(${name} PRIVATE "LINKER:--disable-new-dtags")
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH   "${_yolocpp_rpath}"
    INSTALL_RPATH "${_yolocpp_rpath}"
  )
endfunction()

yolocpp_tool(convert_weights)
