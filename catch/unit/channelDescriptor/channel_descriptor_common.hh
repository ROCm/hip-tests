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

enum class ChannelDimension {
  OneDim,
  TwoDim,
  ThreeDim,
  FourDim
};

template<typename T> class ChannelDescriptorTestShell { 
 protected:
  int size;
  hipChannelFormatKind kind;
  ChannelDimension dimension;
  virtual void SetSizeAndKind() = 0;

 public:
  void Run() {
    hipChannelFormatDesc channel_desc{};
    SetSizeAndKind();
    hipChannelFormatDesc referent_channel_desc{0, 0, 0, 0, kind};
    switch (dimension) {
      case ChannelDimension::FourDim:
        referent_channel_desc.w = size;
      case ChannelDimension::ThreeDim:
        referent_channel_desc.z = size;
      case ChannelDimension::TwoDim:
        referent_channel_desc.y = size;
      default:
        referent_channel_desc.x = size;
    }
    channel_desc = hipCreateChannelDesc<T>();
    REQUIRE(channel_desc.x == referent_channel_desc.x);
    REQUIRE(channel_desc.y == referent_channel_desc.y);
    REQUIRE(channel_desc.z == referent_channel_desc.z);
    REQUIRE(channel_desc.w == referent_channel_desc.w);
    REQUIRE(channel_desc.f == referent_channel_desc.f);
  }

  ChannelDescriptorTestShell(const ChannelDimension dimension): size(0), kind(hipChannelFormatKindNone), dimension(dimension) {}
  ChannelDescriptorTestShell(const ChannelDescriptorTestShell&) = delete;
  ChannelDescriptorTestShell(ChannelDescriptorTestShell&&) = delete;
};

template<typename T> class ChannelDescriptorTest1D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest1D() : ChannelDescriptorTestShell<T>(ChannelDimension::OneDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same<char, T>::value) {
      std::cout << "1" << std::endl;
      this->size = static_cast<int>(sizeof(char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<unsigned char, T>::value || std::is_same<uchar1, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<signed char, T>::value || std::is_same<char1, T>::value) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<unsigned short, T>::value || std::is_same<ushort1, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<short, T>::value || std::is_same<signed short, T>::value ||
               std::is_same<short1, T>::value) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<unsigned int, T>::value || std::is_same<uint1, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<int, T>::value || std::is_same<signed int, T>::value ||
               std::is_same<int1, T>::value) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<float, T>::value || std::is_same<float1, T>::value) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    } else if (std::is_same<unsigned long, T>::value || std::is_same<ulong1, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<long, T>::value || std::is_same<signed long, T>::value ||
               std::is_same<long1, T>::value) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
  }
};

template<typename T> class ChannelDescriptorTest2D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest2D() : ChannelDescriptorTestShell<T>(ChannelDimension::TwoDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same<uchar2, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<char2, T>::value) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;  
    } else if (std::is_same<ushort2, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<short2, T>::value) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<uint2, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<int2, T>::value) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<float2, T>::value) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    } else if (std::is_same<ulong2, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<long2, T>::value) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
  }
};

#ifndef __GNUC__
template<typename T> class ChannelDescriptorTest3D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest3D() : ChannelDescriptorTestShell<T>(ChannelDimension::ThreeDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same<uchar3, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<char3, T>::value) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;  
    } else if (std::is_same<ushort3, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<short3, T>::value) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<uint3, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<int3, T>::value) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<float3, T>::value) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    } else if (std::is_same<ulong3, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<long3, T>::value) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
  }
};
#endif

template<typename T> class ChannelDescriptorTest4D : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTest4D() : ChannelDescriptorTestShell<T>(ChannelDimension::FourDim) {}

 protected:
  void SetSizeAndKind() {
    if (std::is_same<uchar4, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned char) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<char4, T>::value) {
      this->size = static_cast<int>(sizeof(signed char) * 8);
      this->kind = hipChannelFormatKindSigned;  
    } else if (std::is_same<ushort4, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned short) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<short4, T>::value) {
      this->size = static_cast<int>(sizeof(signed short) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<uint4, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned int) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<int4, T>::value) {
      this->size = static_cast<int>(sizeof(signed int) * 8);
      this->kind = hipChannelFormatKindSigned;
    } else if (std::is_same<float4, T>::value) {
      this->size = static_cast<int>(sizeof(float) * 8);
      this->kind = hipChannelFormatKindFloat;
    } else if (std::is_same<ulong4, T>::value) {
      this->size = static_cast<int>(sizeof(unsigned long) * 8);
      this->kind = hipChannelFormatKindUnsigned;
    } else if (std::is_same<long4, T>::value) {
      this->size = static_cast<int>(sizeof(signed long) * 8);
      this->kind = hipChannelFormatKindSigned;
    }
  }
};

template<typename T> class ChannelDescriptorTestNone : public ChannelDescriptorTestShell<T> {
 public:
  ChannelDescriptorTestNone() : ChannelDescriptorTestShell<T>(ChannelDimension::OneDim) {}

 protected:
  void SetSizeAndKind() {}
};
