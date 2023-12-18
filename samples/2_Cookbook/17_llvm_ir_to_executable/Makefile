# Copyright (c) 2020 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
ifeq ($(OS),Windows_NT)
 $(error Makefile is not supported on windows platform. Please use cmake instead to build sample.)
endif
ROCM_PATH?= $(wildcard /opt/rocm/)
HIP_PATH?= $(ROCM_PATH)

HIPCC=$(HIP_PATH)/bin/hipcc
CLANG=$(HIP_PATH)/llvm/bin/clang
LLVM_MC=$(HIP_PATH)/llvm/bin/llvm-mc
CLANG_OFFLOAD_BUNDLER=$(HIP_PATH)/llvm/bin/clang-offload-bundler
LLVM_AS=$(HIP_PATH)/llvm/bin/llvm-as
LLVM_DIS=$(HIP_PATH)/llvm/bin/llvm-dis
INCLUDES  := -I../../common

SRCS=square.cpp

# Extracting the IR code, then creating an executable with the modified IR.

SQ_HOST_BC=square_host.bc
SQ_HOST_LL=square_host.ll
SQ_HOST_OBJ=square_host.o
SQ_DEVICE_OBJ=square_device.o
SQ_DEVICE_HIPFB=offload_bundle.hipfb
SQ_IR_EXE=square_ir.out

MCIN_OBJ_GEN=hip_obj_gen.mcin
GPU_ARCH1=gfx900
GPU_ARCH2=gfx906
GPU_ARCH3=gfx908
GPU_ARCH4=gfx1010
GPU_ARCH5=gfx1030
GPU_ARCH6=gfx1100
GPU_ARCH7=gfx1101
GPU_ARCH8=gfx1102
GPU_ARCH9=gfx1103

.PHONY: test

all: src_to_ir bc_to_ll ll_to_bc ir_to_exec

src_to_ir:
	$(HIPCC) $(INCLUDES) -c -emit-llvm --cuda-host-only -target x86_64-linux-gnu -o $(SQ_HOST_BC) $(SRCS)
	$(HIPCC) $(INCLUDES) -c -emit-llvm --cuda-device-only --offload-arch=$(GPU_ARCH1) --offload-arch=$(GPU_ARCH2) --offload-arch=$(GPU_ARCH3) --offload-arch=$(GPU_ARCH4) --offload-arch=$(GPU_ARCH5) --offload-arch=$(GPU_ARCH6) --offload-arch=$(GPU_ARCH7) --offload-arch=$(GPU_ARCH8) --offload-arch=$(GPU_ARCH9) $(SRCS)

# By default, the LLVM IR Bitcode file names will be:
#   square-hip-amdgcn-amd-amdhsa-gfx900.bc
#   square-hip-amdgcn-amd-amdhsa-gfx906.bc
#   square-hip-amdgcn-amd-amdhsa-gfx908.bc
#   square-hip-amdgcn-amd-amdhsa-gfx1010.bc
#   square-hip-amdgcn-amd-amdhsa-gfx1030.bc

bc_to_ll:
	$(LLVM_DIS) $(SQ_HOST_BC) -o $(SQ_HOST_LL)
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).ll

# You may modify the .ll LLVM IR files before the next step
#
# Note: hipcc does not work to convert .bc to .o, use clang instead.

ll_to_bc:
	$(LLVM_AS) $(SQ_HOST_LL) -o $(SQ_HOST_BC)
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).bc

ir_to_exec:
	$(HIPCC) -c $(SQ_HOST_BC) -o $(SQ_HOST_OBJ)
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH1) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH2) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH3) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH4) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH5) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH6) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH7) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH8) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH9) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).o
	$(CLANG_OFFLOAD_BUNDLER) -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-$(GPU_ARCH1),hip-amdgcn-amd-amdhsa-$(GPU_ARCH2),hip-amdgcn-amd-amdhsa-$(GPU_ARCH3),hip-amdgcn-amd-amdhsa-$(GPU_ARCH4),hip-amdgcn-amd-amdhsa-$(GPU_ARCH5),hip-amdgcn-amd-amdhsa-$(GPU_ARCH6),hip-amdgcn-amd-amdhsa-$(GPU_ARCH7),hip-amdgcn-amd-amdhsa-$(GPU_ARCH8),hip-amdgcn-amd-amdhsa-$(GPU_ARCH9) -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH3).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH4).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH5).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH6).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH7).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH8).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH9).o -outputs=$(SQ_DEVICE_HIPFB)
	$(LLVM_MC) $(MCIN_OBJ_GEN) -o $(SQ_DEVICE_OBJ) --filetype=obj
	$(HIPCC) $(SQ_HOST_OBJ) $(SQ_DEVICE_OBJ) -o $(SQ_IR_EXE)

clean:
	rm -f *.o *.out *.hipfb *.s *.ll *.bc

