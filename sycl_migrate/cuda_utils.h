#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>

#if defined(__HIPCC__)
  #define HOST_DEVICE_INLINE __host__ __device__
  #define DEVICE_INLINE __device__
  #define HOST_INLINE __host__
#elif defined(SYCL_LANGUAGE_VERSION) || defined(_NVHPC_CUDA)
  #define HOST_DEVICE_INLINE __dpct_inline__
  #define DEVICE_INLINE __dpct_inline__
  #define HOST_INLINE __dpct_inline__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
  #define HOST_INLINE inline
#endif

/*
DPCT1009:307: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUDA_CHECK(cmd) \
  do {                  \
    dpct::err0 e = cmd; \
                        \
  } while (0)

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

namespace cuda_utils {

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

};  // namespace cuda_utils