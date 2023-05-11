/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_process.hh>

TEST_CASE("Unit_printf_length") {
#if HT_NVIDIA
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
)here");
#else
  std::string reference(R"here(-42 -42
-42 -42
-42 -42
42 52
42 52
42 52
2a 2A
2a 2A
2a 2A
123.456000
x
123.456000
-42 -42
-42 -42
-42 -42
0 0
42 52
42 52
42 52
0 0
)here");
#endif

  hip::SpawnProc proc("printfLength_exe", true);
  REQUIRE(0 == proc.run());
  REQUIRE(proc.getOutput() == reference);
}
