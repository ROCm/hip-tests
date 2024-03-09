if(NOT RTC_TESTING)
    set(TEST_SRC
        Set.cpp
    )

    add_library(KERNELS EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
    target_compile_options(KERNELS PUBLIC -std=c++17)
    
    # If compiling as part of CHIP-SPV, add the necessary offload flags which would normally be added by hipcc
    if(HIP_PLATFORM STREQUAL "spirv" AND NOT MASTER_PROJECT)
        target_compile_options(KERNELS PUBLIC ${HIP_OFFLOAD_COMPILE_OPTIONS_BUILD_})
    endif()
endif()
