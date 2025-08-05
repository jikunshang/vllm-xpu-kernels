#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuda_utils.h"
#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif

int64_t get_device_attribute(int64_t attribute, int64_t device_id) {
  // Return the cached value on subsequent calls
  static int value = [=]() {
    int device = static_cast<int>(device_id);
    if (device < 0) {
      CUDA_CHECK(DPCT_CHECK_ERROR(device = dpct::get_current_device_id()));
    }
    int value;
    /*
    DPCT1076:24: The device attribute was not recognized. You may need to adjust
    the code.
    */
    CUDA_CHECK(
        cudaDeviceGetAttribute(&value, static_cast<int>(attribute), device));
    return static_cast<int>(value);
  }();

  return value;
}

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id) {
  int64_t attribute;
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
  // cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 if not is_hip() else 74

#ifdef USE_ROCM
  attribute = hipDeviceAttributeMaxSharedMemoryPerBlock;
#else
  attribute = 97;
#endif

  return get_device_attribute(attribute, device_id);
}
