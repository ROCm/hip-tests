## What is this repository for? ###

This repository provides unit tests for  [HIP](https://github.com/ROCm-Developer-Tools/HIP) implementation.

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

©2022 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches:

The hip-tests repository maintains several branches. The branches that are of importance are:

* Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest release is rocm-5.4, main branch will be the repository based on this release.
* Develop branch: This is the default branch, on which the new features are still under development and visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.
* Release branches. These are branches corresponding to each ROCM release, listed with release tags, such as rocm-5.4, etc.

## Release tagging:

hip-tests releases are typically naming convention for each ROCM release to help differentiate them.

* rocm x.yy: These are the stable releases based on the ROCM release.
  This type of release is typically made once a month.*


### Build HIP catch tests

For building HIP from sources, please check instructions on [HIP page] (https://github.com/ROCm-Developer-Tools/HIP/blob/rocm-5.4.x/docs/markdown/hip_build.md#build-hip-on-amd-platform)

HIP catch tests can be built via the following instructions,

Clone the hip-tests from rocm-5.4.x branch
```
git clone -b rocm-5.4.x https://github.com/ROCm-Developer-Tools/hip-tests.git
export HIP_TESTS_DIR="$(readlink -f hip-tests)"
```

Build the catch tests
```
cd "$HIP_TESTS_DIR"
mkdir -p build; cd build
export HIP_PATH=/opt/rocm-5.4/ (or any custom path where HIP is installed)
cmake ../catch/  -DHIP_PLATFORM=amd
make -j$(nproc) build_tests
ctest # run tests
```

HIP catch tests are built under the folder $HIP_TESTS_DIR/build.

### Build HIP Catch2 standalone test

HIP Catch2 supports build a standalone test, for example,

```
hipcc $HIP_TESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc -I ./catch/include ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 -o hipPointerGetAttributes
./hipPointerGetAttributes
```

