/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _qdq_8_cuh
#define _qdq_8_cuh

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "qdq_util.dp.hpp"

namespace vllm {
namespace gptq {

__dpct_inline__ void shuffle_8bit_4(uint32_t* q, int stride) {}

__dpct_inline__ void dequant_8bit_8(const uint32_t q_0, const uint32_t q_1,
                                    sycl::half2 (&dq)[4], int stride,
                                    const uint32_t zero) {
  sycl::half dqh[8];
  for (int i = 0; i < 4; i++) dqh[i] = dq_ns(exb(q_0, i * 8, 0xff), zero);
  for (int i = 0; i < 4; i++) dqh[i + 4] = dq_ns(exb(q_1, i * 8, 0xff), zero);

  for (int i = 0; i < 4; i++)
    dq[i] = sycl::half2(dqh[i * 2], dqh[i * 2 + 1]);
}

}  // namespace gptq
}  // namespace vllm

#endif
