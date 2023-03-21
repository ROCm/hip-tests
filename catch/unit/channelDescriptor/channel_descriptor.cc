/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "channel_descriptor_common.hh"

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_1D", "", char, unsigned char, uchar1,
                   signed char, char1, unsigned short, ushort1, signed short, short1,
                   unsigned int, uint1, signed int, int1, float, float1, unsigned long, ulong1,
                   signed long, long1) {
  ChannelDescriptorTest1D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_2D", "", uchar2, char2, ushort2, short2,
                   uint2, int2, float2, ulong2, long2) {
  ChannelDescriptorTest2D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

#ifndef __GNUC__
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_3D", "", uchar3, char3, ushort3, short3,
                   uint3, int3, float3, ulong3, long3) {
  ChannelDescriptorTest3D<TestType> channel_desc_test;
  channel_desc_test.Run();
}
#endif

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_4D", "", uchar4, char4, ushort4, short4,
                   uint4, int4, float4, ulong4, long4) {
  ChannelDescriptorTest4D<TestType> channel_desc_test;
  channel_desc_test.Run();
}

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_FormatNone", "", short, int, long, long long,
                   signed long long, unsigned long long) {
  ChannelDescriptorTestNone<TestType> channel_desc_test;
  channel_desc_test.Run();
}
