if(NOT RTC_TESTING)
    set(TEST_SRC
        Set.cpp
    )

    # TODO: check exactly what it does
    add_library(KERNELS EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
    target_compile_options(KERNELS PUBLIC -std=c++17)
endif()
