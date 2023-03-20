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

#include <hip_test_common.hh>

static void CompareChannelDescriptors(const hipChannelFormatDesc& lchannel_desc, const hipChannelFormatDesc& rchannel_desc) {
  REQUIRE(lchannel_desc.x == rchannel_desc.x);
  REQUIRE(lchannel_desc.y == rchannel_desc.y);
  REQUIRE(lchannel_desc.z == rchannel_desc.z);
  REQUIRE(lchannel_desc.w == rchannel_desc.w);
  REQUIRE(lchannel_desc.f == rchannel_desc.f);
}

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_1D", "", char, unsigned char, uchar1,
                   signed char, char1, unsigned short, ushort1, signed short, short1,
                   unsigned int, uint1, signed int, int1, float, float1, unsigned long, ulong1,
                   signed long, long1) {
  hipChannelFormatDesc channel_desc{};
  int size{};
  hipChannelFormatKind kind{};

  if (std::is_same<char, TestType>::value) {
    size = static_cast<int>(sizeof(char) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<unsigned char, TestType>::value || std::is_same<uchar1, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned char) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<signed char, TestType>::value || std::is_same<char1, TestType>::value) {
    size = static_cast<int>(sizeof(signed char) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<unsigned short, TestType>::value || std::is_same<ushort1, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned short) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<signed short, TestType>::value || std::is_same<short1, TestType>::value) {
    size = static_cast<int>(sizeof(signed short) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<unsigned int, TestType>::value || std::is_same<uint1, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned int) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<signed int, TestType>::value || std::is_same<int1, TestType>::value) {
    size = static_cast<int>(sizeof(signed int) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<float, TestType>::value || std::is_same<float1, TestType>::value) {
    size = static_cast<int>(sizeof(float) * 8);
    kind = hipChannelFormatKindFloat;
  } else if (std::is_same<unsigned long, TestType>::value || std::is_same<ulong1, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned long) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<signed long, TestType>::value || std::is_same<long1, TestType>::value) {
    size = static_cast<int>(sizeof(signed long) * 8);
    kind = hipChannelFormatKindSigned;
  }

  const hipChannelFormatDesc referent_channel_desc{size, 0, 0, 0, kind};
  channel_desc = hipCreateChannelDesc<TestType>();
  CompareChannelDescriptors(channel_desc, referent_channel_desc);
}

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_2D", "", uchar2, char2, ushort2, short2,
                   uint2, int2, float2, ulong2, long2) {
  hipChannelFormatDesc channel_desc{};
  int size{};
  hipChannelFormatKind kind{};

  if (std::is_same<uchar2, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned char) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<char2, TestType>::value) {
    size = static_cast<int>(sizeof(signed char) * 8);
    kind = hipChannelFormatKindSigned;  
  } else if (std::is_same<ushort2, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned short) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<short2, TestType>::value) {
    size = static_cast<int>(sizeof(signed short) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<uint2, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned int) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<int2, TestType>::value) {
    size = static_cast<int>(sizeof(signed int) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<float2, TestType>::value) {
    size = static_cast<int>(sizeof(float) * 8);
    kind = hipChannelFormatKindFloat;
  } else if (std::is_same<ulong2, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned long) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<long2, TestType>::value) {
    size = static_cast<int>(sizeof(signed long) * 8);
    kind = hipChannelFormatKindSigned;
  }

  const hipChannelFormatDesc referent_channel_desc{size, size, 0, 0, kind};
  channel_desc = hipCreateChannelDesc<TestType>();
  CompareChannelDescriptors(channel_desc, referent_channel_desc);
}

#ifndef __GNUC__
TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_3D", "", uchar3, char3, ushort3, short3,
                   uint3, int3, float3, ulong3, long3) {
  hipChannelFormatDesc channel_desc{};
  int size{};
  hipChannelFormatKind kind{};

  if (std::is_same<uchar3, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned char) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<char3, TestType>::value) {
    size = static_cast<int>(sizeof(signed char) * 8);
    kind = hipChannelFormatKindSigned;  
  } else if (std::is_same<ushort3, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned short) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<short3, TestType>::value) {
    size = static_cast<int>(sizeof(signed short) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<uint3, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned int) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<int3, TestType>::value) {
    size = static_cast<int>(sizeof(signed int) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<float3, TestType>::value) {
    size = static_cast<int>(sizeof(float) * 8);
    kind = hipChannelFormatKindFloat;
  } else if (std::is_same<ulong3, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned long) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<long3, TestType>::value) {
    size = static_cast<int>(sizeof(signed long) * 8);
    kind = hipChannelFormatKindSigned;
  }

  const hipChannelFormatDesc referent_channel_desc{size, size, size, 0, kind};
  channel_desc = hipCreateChannelDesc<TestType>();
  CompareChannelDescriptors(channel_desc, referent_channel_desc);
}
#endif

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_Basic_4D", "", uchar4, char4, ushort4, short4,
                   uint4, int4, float4, ulong4, long4) {
  hipChannelFormatDesc channel_desc{};
  int size{};
  hipChannelFormatKind kind{};

  if (std::is_same<uchar4, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned char) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<char4, TestType>::value) {
    size = static_cast<int>(sizeof(signed char) * 8);
    kind = hipChannelFormatKindSigned;  
  } else if (std::is_same<ushort4, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned short) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<short4, TestType>::value) {
    size = static_cast<int>(sizeof(signed short) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<uint4, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned int) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<int4, TestType>::value) {
    size = static_cast<int>(sizeof(signed int) * 8);
    kind = hipChannelFormatKindSigned;
  } else if (std::is_same<float4, TestType>::value) {
    size = static_cast<int>(sizeof(float) * 8);
    kind = hipChannelFormatKindFloat;
  } else if (std::is_same<ulong4, TestType>::value) {
    size = static_cast<int>(sizeof(unsigned long) * 8);
    kind = hipChannelFormatKindUnsigned;
  } else if (std::is_same<long4, TestType>::value) {
    size = static_cast<int>(sizeof(signed long) * 8);
    kind = hipChannelFormatKindSigned;
  }

  const hipChannelFormatDesc referent_channel_desc{size, size, size, size, kind};
  channel_desc = hipCreateChannelDesc<TestType>();
  CompareChannelDescriptors(channel_desc, referent_channel_desc);
}

TEMPLATE_TEST_CASE("Unit_ChannelDescriptor_Positive_FormatNone", "",short, int, long, long long,
          signed long long, unsigned long long) {
  hipChannelFormatDesc channel_desc{};
  const hipChannelFormatDesc referent_channel_desc{0, 0, 0, 0, hipChannelFormatKindNone};
  channel_desc = hipCreateChannelDesc<TestType>();
  CompareChannelDescriptors(channel_desc, referent_channel_desc);
}
