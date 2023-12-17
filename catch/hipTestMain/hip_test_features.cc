#include "hip_test_features.hh"

#include <iostream>
#include <assert.h>

#include "hip_test_context.hh"

std::vector<std::unordered_set<std::string>> GCNArchFeatMap = {
  {"gfx90a", "gfx940", "gfx941", "gfx942"},               // CT_FEATURE_FINEGRAIN_HWSUPPORT
  {"gfx90a", "gfx940", "gfx941", "gfx942"},               // CT_FEATURE_HMM
  {"gfx90a", "gfx940", "gfx941", "gfx942"},               // CT_FEATURE_TEXTURES_NOT_SUPPORTED
};

#if HT_AMD
std::string TrimAndGetGFXName(const std::string& full_gfx_name) {
  std::string gfx_name("");

  // Split the first part of the delimiter
  std::string delimiter = ":";
  auto pos = full_gfx_name.find(delimiter);
  if (pos == std::string::npos) {
    gfx_name = full_gfx_name;
  } else {
    gfx_name = full_gfx_name.substr(0, pos);
  }

  assert(gfx_name.substr(0,3) == "gfx");
  return gfx_name;
}
#endif

// Check if the GCN Maps
bool CheckIfFeatSupported(enum CTFeatures test_feat, std::string gcn_arch) {
#if HT_NVIDIA
  return true; // returning true since feature check does not exist for NV. 
#elif HT_AMD
  assert(test_feat >= 0 && test_feat < CTFeatures::CT_FEATURE_LAST);
  gcn_arch = TrimAndGetGFXName(gcn_arch);
  assert(gcn_arch != "");
  return (GCNArchFeatMap[test_feat].find(gcn_arch) != GCNArchFeatMap[test_feat].cend());
#else
  std::cout<<"Platform has to be either AMD or NVIDIA, asserting..."<<std::endl;
  assert(false);
#endif
}
