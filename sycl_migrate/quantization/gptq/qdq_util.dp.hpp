#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_util_cuh
#define _qdq_util_cuh

namespace vllm {
namespace gptq {

union half2_uint32 {
  uint32_t as_uint32;
  sycl::half2 as_half2{};
  half2_uint32(uint32_t val) : as_uint32(val) {}
  half2_uint32(sycl::half2 val) : as_half2(val) {}
};

union half_uint16 {
  uint16_t as_uint16;
  sycl::half as_half{};
  half_uint16(uint16_t val) : as_uint16(val) {}
  half_uint16(sycl::half val) : as_half(val) {}
};

// Max_scale premultiplied by 1/256

__dpct_inline__ sycl::half dq_scale(const int qs, const sycl::half max_scale) {
  int qs_i = qs + 1;
  sycl::half qs_h = sycl::vec<int, 1>(qs_i * qs_i)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  qs_h = qs_h * max_scale;
  return qs_h;
}

__dpct_inline__ sycl::half dq(const int q, const int qzero,
                              const sycl::half scale) {
  return sycl::vec<int, 1>(q - qzero)
             .convert<sycl::half, sycl::rounding_mode::rte>()[0] *
         scale;
}

__dpct_inline__ sycl::half dq_ns(const int q, const int qzero) {
  // return __hsub(__int2half_rn(q), __int2half_rn(qzero));
  return sycl::vec<int, 1>(q - qzero)
      .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}

__dpct_inline__ int exb(const uint32_t q, const int shift, const int mask) {
  return (int)((q >> shift) & mask);
}

__dpct_inline__ int exb(const uint32_t q1, const uint32_t q0, const int shift,
                        const int mask) {
  /*
  DPCT1017:356: The dpct::funnelshift_rc call is used instead of the
  __funnelshift_rc call. These two calls do not provide exactly the same
  functionality. Check the potential precision and/or performance issues for the
  generated code.
  */
  return (int)(dpct::funnelshift_rc(q0, q1, shift) & mask);
}

}  // namespace gptq
}  // namespace vllm
#endif
