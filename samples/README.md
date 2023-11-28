Build procedure

The CMakeLists.txt at hip-tests/samples folder is currently used for packaging purpose.
We provide Makefile and CMakeLists.txt to build the samples seperately.

1.Makefile supports shared lib of hip-rocclr runtime and nvcc.

To build a sample, just type in sample folder,

make



2.CMakeLists.txt can support shared and static libs of hip-rocclr runtime.
The same steps can be followed for both.

To build a sample, run in the sample folder,

mkdir -p build && cd build

rm -rf * (to clear up)

cmake -DCMAKE_PREFIX_PATH=<path/to/rocm> -DHIP_CXX_COMPILER=<path/to/clang> ..

Then run,

make

Note that if you want debug version, add "-DCMAKE_BUILD_TYPE=Debug" in cmake cmd.


3.To package samples and generate packages. From hip-tests/build

cmake ../samples

make package_samples

