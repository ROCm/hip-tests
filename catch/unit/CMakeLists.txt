# Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(rtc)
endif()
add_subdirectory(deviceLib)
add_subdirectory(graph)
add_subdirectory(memory)
add_subdirectory(stream_ordered)
add_subdirectory(stream)
add_subdirectory(event)
add_subdirectory(device)
add_subdirectory(printf)
add_subdirectory(streamperthread)
add_subdirectory(kernel)
add_subdirectory(multiThread)
add_subdirectory(compiler)
add_subdirectory(errorHandling)
add_subdirectory(cooperativeGrps)
add_subdirectory(context)
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(device_memory)  # device memcpy/memset not supported on SPIR-V
  add_subdirectory(warp)  # shfl ambiguity on SPIR-V
endif()
add_subdirectory(dynamicLoading)
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(g++)  # requires AMD host_defines.h not available in chipStar
endif()
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(module)  # uses --genco (AMD-specific)
endif()
add_subdirectory(channelDescriptor)
add_subdirectory(executionControl)
add_subdirectory(math)
add_subdirectory(vector_types)
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(atomics)  # system/safe/unsafe atomics not available on SPIR-V
endif()
add_subdirectory(complex)
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(p2p)  # hipExtGetLinkTypeAndHopCount not in chipStar
endif()
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(gcc)  # requires AMD headers not available in chipStar
endif()
add_subdirectory(syncthreads)
add_subdirectory(threadfence)
if(NOT HIP_PLATFORM_IS_SPIRV)
  add_subdirectory(virtualMemoryManagement)  # hipMemSetAccess not in chipStar
endif()


if(NOT HIP_PLATFORM_IS_SPIRV)
    add_subdirectory(occupancy)
    add_subdirectory(surface)
    add_subdirectory(texture)
endif()

if(HIP_PLATFORM STREQUAL "amd" AND NOT HIP_PLATFORM_IS_SPIRV)
add_subdirectory(callback)
#add_subdirectory(clock)
add_subdirectory(hip_specific)
# Vulkan interop APIs currently undefined for Nvidia
add_subdirectory(vulkan_interop)
add_subdirectory(gl_interop) # Disabled on NVIDIA due to defect - EXSWHTEC-246
endif()
add_subdirectory(synchronization)
add_subdirectory(launchBounds)
add_subdirectory(assertion)