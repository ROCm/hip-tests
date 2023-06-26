set(COMMON_SHARED_SRC vulkan_test.cc)

set(TEST_SRC
    hipExternalMemoryGetMappedBuffer.cc
    hipImportExternalMemory.cc
    hipDestroyExternalMemory.cc
    hipWaitExternalSemaphoresAsync.cc
    hipSignalExternalSemaphoresAsync.cc
    hipImportExternalSemaphore.cc
    hipDestroyExternalSemaphore.cc
   )

find_package(Vulkan)
if(NOT Vulkan_FOUND)
  if(EXISTS "${VULKAN_PATH}")
    message(STATUS "Vulkan SDK: ${VULKAN_PATH}")
  elseif (EXISTS "$ENV{VULKAN_SDK}")
    message(STATUS "FOUND VULKAN SDK: $ENV{VULKAN_SDK}")
    set(VULKAN_PATH  $ENV{VULKAN_SDK})
  else()
    message(STATUS "Vulkan sdk not found, interop test not enabled. Please add VULKAN_PATH to enable the test")
    return()
  endif()
endif()

if(WIN32)
  set(LINKER_LIBS vulkan-1)
else()
  set(LINKER_LIBS vulkan)
endif()

hip_add_exe_to_target(NAME VulkanInteropTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
                      LINKER_LIBS ${LINKER_LIBS})
