# AMD Tests
set(TEST_SRC
    printfFlags.cc
    printfSpecifiers.cc
    printfHost.cc
)

if(UNIX)
  set(AMD_TEST_SRC
      printfNonHost.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()

if(UNIX)
  if(HIP_PLATFORM MATCHES "amd")
    set_source_files_properties(printfNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
    set_source_files_properties(printfFlagsNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
  endif()
endif()

if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME printfTests
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    PROPERTY CXX_STANDARD 17)
elseif (HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME printfTests
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    COMPILE_OPTIONS -std=c++17)
endif()

# Standalone exes
add_executable(printfFlags_exe EXCLUDE_FROM_ALL printfFlags_exe.cc)
add_executable(printfSpecifiers_exe EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)

add_dependencies(build_tests printfFlags_exe)
add_dependencies(build_tests printfSpecifiers_exe)
