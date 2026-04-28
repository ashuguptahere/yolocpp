function(yolocpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE yolocpp_core)
  target_link_options(${name} PRIVATE "LINKER:--disable-new-dtags")
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    BUILD_RPATH   "${_yolocpp_rpath}"
    INSTALL_RPATH "${_yolocpp_rpath}"
  )
  add_test(NAME ${name} COMMAND ${name})
endfunction()

yolocpp_add_test(smoke_test         smoke_test.cpp smoke_kernel.cu)
yolocpp_add_test(test_yolo8_arch   test_yolo8_arch.cpp)
yolocpp_add_test(test_pt_loader     test_pt_loader.cpp)
yolocpp_add_test(test_predict       test_predict.cpp)
yolocpp_add_test(test_train_overfit test_train_overfit.cpp)
yolocpp_add_test(test_trt_export    test_trt_export.cpp)
yolocpp_add_test(test_benchmark     test_benchmark.cpp)
yolocpp_add_test(test_phase3_tasks  test_phase3_tasks.cpp)
yolocpp_add_test(test_phase31_train test_phase31_train.cpp)
yolocpp_add_test(test_pt_save       test_pt_save.cpp)
yolocpp_add_test(test_all_scales    test_all_scales.cpp)
yolocpp_add_test(test_ddp_smoke     test_ddp_smoke.cpp)
yolocpp_add_test(test_v3_v5         test_v3_v5.cpp)
set_tests_properties(test_v3_v5 PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_pt_loader test_predict test_train_overfit
                     test_trt_export test_benchmark test_phase3_tasks
                     test_phase31_train test_pt_save test_all_scales
                     PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_phase31_train PROPERTIES TIMEOUT 600)
# TRT engine build can take a couple minutes on first run.
set_tests_properties(test_trt_export test_benchmark PROPERTIES TIMEOUT 600)
