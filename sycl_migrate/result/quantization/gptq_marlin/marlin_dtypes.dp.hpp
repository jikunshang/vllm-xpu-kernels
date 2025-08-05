
#ifndef _data_types_cuh
#define _data_types_cuh
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "marlin.dp.hpp"

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<sycl::half> {
 public:
  using scalar_t = sycl::half;
  using scalar_t2 = sycl::half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<sycl::half2, 4>;
  using FragB = Vec<sycl::half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<sycl::half2, 1>;
  using FragZP = Vec<sycl::half2, 4>;

  static float inline num2float(const sycl::half x) {
    return sycl::vec<sycl::half, 1>(x)
        .convert<float, sycl::rounding_mode::automatic>()[0];
  }

  static sycl::half2 inline num2num2(const sycl::half x) {
    return sycl::half2(x);
  }

  static sycl::half2 inline nums2num2(const sycl::half x1,
                                      const sycl::half x2) {
    return sycl::half2(x1, x2);
  }

  static sycl::half inline float2num(const float x) {
    return sycl::vec<float, 1>(x)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
};

template <>
class ScalarType<sycl::ext::oneapi::bfloat16> {
 public:
  using scalar_t = sycl::ext::oneapi::bfloat16;
  using scalar_t2 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>;

  using FragA = Vec<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, 4>;
  using FragB = Vec<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, 1>;
  using FragZP = Vec<sycl::marray<sycl::ext::oneapi::bfloat16, 2>, 4>;

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 800
  static float inline num2float(const sycl::ext::oneapi::bfloat16 x) {
    return static_cast<float>(x);
  }

  static sycl::marray<sycl::ext::oneapi::bfloat16, 2> inline num2num2(
      const sycl::ext::oneapi::bfloat16 x) {
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x, x);
  }

  static sycl::marray<sycl::ext::oneapi::bfloat16, 2> inline nums2num2(
      const sycl::ext::oneapi::bfloat16 x1,
      const sycl::ext::oneapi::bfloat16 x2) {
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x1, x2);
  }

  static sycl::ext::oneapi::bfloat16 inline float2num(const float x) {
    return sycl::ext::oneapi::bfloat16(x);
  }
#endif
};

}  // namespace MARLIN_NAMESPACE_NAME

#endif
