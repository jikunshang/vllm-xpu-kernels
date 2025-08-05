/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_4_cuh
#define _qdq_4_cuh

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "qdq_util.dp.hpp"

namespace vllm {
namespace gptq {
// Permutation:
//
// 77775555 33331111  66664444 22220000

__dpct_inline__ void shuffle_4bit_8(uint32_t* q, int stride) {
  uint32_t qa = q[0];
  uint32_t qb = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0f;
    uint32_t qa1 = (qa & 0xf0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

__dpct_inline__ void dequant_4bit_8(const uint32_t q_0, sycl::half2 (&dq)[4],
                                    int stride, const uint32_t zero) {
  const uint32_t c0 = 0x64006400;
  const sycl::half y16_ =
      sycl::vec<float, 1>(1.0f / 16.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half2 y16 = sycl::half2(y16_, y16_);
  const half_uint16 z1_(0xe400 | zero);  // half(-1024.0f - zero);
  const sycl::half z16_ =
      sycl::vec<int, 1>(-64)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
      sycl::vec<int, 1>(zero)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half2 z1 = sycl::half2(z1_.as_half);
  const sycl::half2 z16 = sycl::half2(z16_);

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x000f000f) | c0);  // half2(q[ 0], q[ 1])      + 1024
  half2_uint32 q1((qa & 0x00f000f0) | c0);  // half2(q[ 2], q[ 3]) * 16 + 1024
  qa >>= 8;
  half2_uint32 q2((qa & 0x000f000f) | c0);  // half2(q[ 4], q[ 5])      + 1024
  half2_uint32 q3((qa & 0x00f000f0) | c0);  // half2(q[ 6], q[ 7]) * 16 + 1024

  dq[0] = q0.as_half2 + z1;
  dq[1] = sycl::fma(q1.as_half2, y16, z16);
  dq[2] = q2.as_half2 + z1;
  dq[3] = sycl::fma(q3.as_half2, y16, z16);
}

__dpct_inline__ void dequant_4bit_8_prep_zero_scale(const uint32_t zero,
                                                    const sycl::half scale,
                                                    sycl::half2 (&z1z16)[2],
                                                    sycl::half2 (&y1y16)[2]) {
  half_uint16 z1(0xe400 | zero);  // half(-1024.0f - zero);
  sycl::half z16 = sycl::vec<int, 1>(-64)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
                   sycl::vec<int, 1>(zero)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0];

  sycl::half2 scale2 = sycl::half2(scale);

  z1z16[0] = scale2 * sycl::half2(z1.as_half);
  z1z16[1] = scale2 * sycl::half2(z16);

  const sycl::half y1 = sycl::vec<float, 1>(1.0f)
                            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half y16 =
      sycl::vec<float, 1>(1.0f / 16.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];

  y1y16[0] = scale2 * sycl::half2(y1);
  y1y16[1] = scale2 * sycl::half2(y16);
}

__dpct_inline__ void dequant_4bit_8_prep_zero(const uint32_t zero,
                                              sycl::half2 (&z1z16)[2],
                                              sycl::half2 (&y1y16)[2]) {
  half_uint16 z1(0xe400 | zero);  // half(-1024.0f - zero);
  sycl::half z16 = sycl::vec<int, 1>(-64)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
                   sycl::vec<int, 1>(zero)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0];

  z1z16[0] = sycl::half2(z1.as_half);
  z1z16[1] = sycl::half2(z16);

  const sycl::half y1 = sycl::vec<float, 1>(1.0f)
                            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half y16 =
      sycl::vec<float, 1>(1.0f / 16.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];

  y1y16[0] = sycl::half2(y1);
  y1y16[1] = sycl::half2(y16);
}

__dpct_inline__ void dequant_4bit_8_gptq(const uint32_t q_0,
                                         sycl::half2 (&dq)[4],
                                         sycl::half2 (&z1z16)[2],
                                         sycl::half2 (&y1y16)[2], int stride,
                                         bool scaled) {
  const uint32_t c0 = 0x64006400;

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x000f000f) |
                  c0);  // half2( q[0]      + 1024, q[1]      + 1024 )
  half2_uint32 q1((qa & 0x00f000f0) |
                  c0);  // half2( q[2] * 16 + 1024, q[3] * 16 + 1024 )
  qa >>= 8;
  half2_uint32 q2((qa & 0x000f000f) |
                  c0);  // half2( q[4]      + 1024, q[5]      + 1024 )
  half2_uint32 q3((qa & 0x00f000f0) |
                  c0);  // half2( q[6] * 16 + 1024, q[7] * 16 + 1024 )

  if (scaled) {
    dq[0] = sycl::fma(q0.as_half2, y1y16[0],
                      z1z16[0]);  // half2( q[0] * s - z * s, q[1] * s - z * s)
    dq[1] = sycl::fma(q1.as_half2, y1y16[1],
                      z1z16[1]);  // half2( q[2] * s - z * s, q[3] * s - z * s)
    dq[2] = sycl::fma(q2.as_half2, y1y16[0], z1z16[0]);
    dq[3] = sycl::fma(q3.as_half2, y1y16[1], z1z16[1]);
  } else {
    dq[0] = q0.as_half2 + z1z16[0];  // half2( q[0] - z, q[1] - z )
    dq[1] = sycl::fma(q1.as_half2, y1y16[1],
                      z1z16[1]);     // half2( q[2] - z, q[3] - z )
    dq[2] = q2.as_half2 + z1z16[0];  // half2( q[4] - z, q[5] - z )
    dq[3] = sycl::fma(q3.as_half2, y1y16[1],
                      z1z16[1]);  // half2( q[6] - z, q[7] - z )
  }
}
}  // namespace gptq
}  // namespace vllm

#endif
