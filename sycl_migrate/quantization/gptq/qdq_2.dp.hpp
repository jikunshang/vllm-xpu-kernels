/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_2_cuh
#define _qdq_2_cuh

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "qdq_util.dp.hpp"

namespace vllm {
namespace gptq {

// Permutation:
//
// ffddbb99 77553311  eeccaa88 66442200

__dpct_inline__ void shuffle_2bit_16(uint32_t* q, int stride) {
  uint32_t qa = q[0];
  uint32_t qb = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t qa0 = qa & 0x03;
    uint32_t qa1 = (qa & 0x0c) >> 2;
    qa >>= 4;
    qb |= (qa1 << (i * 2 + 16));
    qb |= (qa0 << (i * 2));
  }
  q[0] = qb;
}

__dpct_inline__ void dequant_2bit_16(const uint32_t q_0, sycl::half2 (&dq)[8],
                                     int stride, const uint32_t zero) {
  const uint32_t c0 = 0x64006400;
  const sycl::half y4_ =
      sycl::vec<float, 1>(1.0f / 4.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half y16_ =
      sycl::vec<float, 1>(1.0f / 16.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half y64_ =
      sycl::vec<float, 1>(1.0f / 64.0f)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half2 y4 = sycl::half2(y4_, y4_);
  const sycl::half2 y16 = sycl::half2(y16_, y16_);
  const sycl::half2 y64 = sycl::half2(y64_, y64_);

  const half_uint16 z1_(0xe400 | zero);  // half(-1024.0f - zero);
  const sycl::half z4_ =
      sycl::vec<int, 1>(-256)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
      sycl::vec<int, 1>(zero)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half z16_ =
      sycl::vec<int, 1>(-64)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
      sycl::vec<int, 1>(zero)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half z64_ =
      sycl::vec<int, 1>(-16)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
      sycl::vec<int, 1>(zero)
          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  const sycl::half2 z1 = sycl::half2(z1_.as_half);
  const sycl::half2 z4 = sycl::half2(z4_);
  const sycl::half2 z16 = sycl::half2(z16_);
  const sycl::half2 z64 = sycl::half2(z64_);

  uint32_t qa = q_0;
  half2_uint32 q0((qa & 0x00030003) | c0);  // half2(q[ 0], q[ 1])      + 1024
  half2_uint32 q1((qa & 0x000c000c) | c0);  // half2(q[ 2], q[ 3]) *  4 + 1024
  half2_uint32 q2((qa & 0x00300030) | c0);  // half2(q[ 4], q[ 5]) * 16 + 1024
  half2_uint32 q3((qa & 0x00c000c0) | c0);  // half2(q[ 6], q[ 7]) * 64 + 1024
  qa >>= 8;
  half2_uint32 q4((qa & 0x00030003) | c0);  // half2(q[ 8], q[ 8])      + 1024
  half2_uint32 q5((qa & 0x000c000c) | c0);  // half2(q[10], q[11]) *  4 + 1024
  half2_uint32 q6((qa & 0x00300030) | c0);  // half2(q[12], q[13]) * 16 + 1024
  half2_uint32 q7((qa & 0x00c000c0) | c0);  // half2(q[14], q[15]) * 64 + 1024

  dq[0] = q0.as_half2 + z1;
  dq[1] = sycl::fma(q1.as_half2, y4, z4);
  dq[2] = sycl::fma(q2.as_half2, y16, z16);
  dq[3] = sycl::fma(q3.as_half2, y64, z64);
  dq[4] = q4.as_half2 + z1;
  dq[5] = sycl::fma(q5.as_half2, y4, z4);
  dq[6] = sycl::fma(q6.as_half2, y16, z16);
  dq[7] = sycl::fma(q7.as_half2, y64, z64);
}

}  // namespace gptq
}  // namespace vllm

#endif
