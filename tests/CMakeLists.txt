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

# Dev-only inspector — built but not registered as a ctest.
add_executable(inspect_pt inspect_pt.cpp)
target_link_libraries(inspect_pt PRIVATE yolocpp_core)
target_link_options(inspect_pt PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(inspect_pt PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

add_executable(diff_model_pt diff_model_pt.cpp)
target_link_libraries(diff_model_pt PRIVATE yolocpp_core)
target_link_options(diff_model_pt PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(diff_model_pt PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

add_executable(run_convert run_convert.cpp)
target_link_libraries(run_convert PRIVATE yolocpp_core)
target_link_options(run_convert PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(run_convert PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

# Dev-only video synthesiser used as the fixture for #51C2 manual
# smokes. Not a ctest.
add_executable(make_test_video make_test_video.cpp)
target_link_libraries(make_test_video PRIVATE yolocpp_core)
target_link_options(make_test_video PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(make_test_video PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

# Per-layer parity comparator. Reads dumps from /tmp/yolocpp_parity/ written
# by the (uncommitted) Python harness; not a ctest.
add_executable(parity_compare parity_compare.cpp)
target_link_libraries(parity_compare PRIVATE yolocpp_core)
target_link_options(parity_compare PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(parity_compare PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

# Yolo10 forward-output dumper for ONNX parity validation. Not a ctest
# (gated on weights availability + invoked by the v10_onnx_parity.py
# script under /tmp/).
add_executable(dump_v10_forward dump_v10_forward.cpp)
target_link_libraries(dump_v10_forward PRIVATE yolocpp_core)
target_link_options(dump_v10_forward PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(dump_v10_forward PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

yolocpp_add_test(smoke_test         smoke_test.cpp smoke_kernel.cu)
yolocpp_add_test(test_registry      test_registry.cpp)
yolocpp_add_test(test_api           test_api.cpp)
set_tests_properties(test_api PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 60)
yolocpp_add_test(test_cross_backend_parity test_cross_backend_parity.cpp)
set_tests_properties(test_cross_backend_parity PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 600)  # TRT engine builds dominate first run; cached afterward
yolocpp_add_test(test_flat_dataset test_flat_dataset.cpp)
set_tests_properties(test_flat_dataset PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 30)
yolocpp_add_test(test_voc_coco_datasets test_voc_coco_datasets.cpp)
set_tests_properties(test_voc_coco_datasets PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 30)
yolocpp_add_test(test_mosaic_mixup test_mosaic_mixup.cpp)
set_tests_properties(test_mosaic_mixup PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 30)
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
yolocpp_add_test(test_v4_v6_v7      test_v4_v6_v7.cpp)
yolocpp_add_test(test_v1_e2e        test_v1_e2e.cpp)
yolocpp_add_test(test_v2_e2e        test_v2_e2e.cpp)
yolocpp_add_test(test_v4_e2e        test_v4_e2e.cpp)
yolocpp_add_test(test_v6_e2e        test_v6_e2e.cpp)
yolocpp_add_test(test_v7_e2e        test_v7_e2e.cpp)
yolocpp_add_test(test_val_v3_v10    test_val_v3_v10.cpp)
yolocpp_add_test(test_v9_train      test_v9_train.cpp)
yolocpp_add_test(test_v3_train      test_v3_train.cpp)
yolocpp_add_test(test_v6_train      test_v6_train.cpp)
yolocpp_add_test(test_v4_v7_train   test_v4_v7_train.cpp)
yolocpp_add_test(test_v10_train     test_v10_train.cpp)
yolocpp_add_test(test_v10_dual_train test_v10_dual_train.cpp)
set_tests_properties(test_v1_e2e test_v2_e2e test_v4_e2e test_v6_e2e
                     test_v7_e2e test_val_v3_v10
                     test_v9_train test_v3_train test_v6_train test_v4_v7_train
                     test_v10_train test_v10_dual_train PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 600)
yolocpp_add_test(test_yolo11        test_yolo11.cpp)
yolocpp_add_test(test_yolo26_train  test_yolo26_train.cpp)
yolocpp_add_test(test_v13_ada       test_v13_ada.cpp)
yolocpp_add_test(test_v13_full      test_v13_full.cpp)
set_tests_properties(test_v13_full PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 120)
add_executable(test_v13_layer_diff test_v13_layer_diff.cpp)
target_link_libraries(test_v13_layer_diff PRIVATE yolocpp_core)
target_link_options(test_v13_layer_diff PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(test_v13_layer_diff PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)
set_tests_properties(test_yolo11 PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 300)
set_tests_properties(test_v3_v5 PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_pt_loader test_predict test_train_overfit
                     test_trt_export test_benchmark test_phase3_tasks
                     test_phase31_train test_pt_save test_all_scales
                     PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(test_phase31_train PROPERTIES TIMEOUT 600)
# TRT engine build can take a couple minutes on first run.
set_tests_properties(test_trt_export test_benchmark PROPERTIES TIMEOUT 600)
yolocpp_add_test(test_v9_e2e        test_v9_e2e.cpp)
set_tests_properties(test_v9_e2e PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 300)
yolocpp_add_test(test_v10_e2e       test_v10_e2e.cpp)
set_tests_properties(test_v10_e2e PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 300)
yolocpp_add_test(test_v3_e2e        test_v3_e2e.cpp)
set_tests_properties(test_v3_e2e PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  TIMEOUT 300)

# Dev-only TRT raw-output dumper for v10 parity debugging. Not a ctest.
add_executable(dump_trt dump_trt.cpp)
target_link_libraries(dump_trt PRIVATE yolocpp_core TensorRT::nvinfer TensorRT::nvinfer_plugin)
target_link_options(dump_trt PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(dump_trt PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)

# Dev-only v6 layer dumper for v6l6 parity debugging. Not a ctest.
add_executable(dump_v6_layers dump_v6_layers.cpp)
target_link_libraries(dump_v6_layers PRIVATE yolocpp_core)
target_link_options(dump_v6_layers PRIVATE "LINKER:--disable-new-dtags")
set_target_properties(dump_v6_layers PROPERTIES
  BUILD_RPATH   "${_yolocpp_rpath}"
  INSTALL_RPATH "${_yolocpp_rpath}"
)
