set(TEST_SRC
    hip_module_common.cc
    hipModuleLoad.cc 
    hipModuleLoadData.cc 
    hipModuleLoadDataEx.cc 
    hipModuleUnload.cc 
)

if(UNIX)
    configure_file(not_a_module.txt not_a_module.txt)

    add_custom_target(empty_module.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/empty_module.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/empty_module.code -I${HIP_PATH}/include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

    if(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME ModuleTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc)
    elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME ModuleTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          COMPILE_OPTIONS --Wno-deprecated-declarations)
    endif()

    add_dependencies(build_tests empty_module.code)
endif()