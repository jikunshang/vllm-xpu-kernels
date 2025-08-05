/*
 * Modified by Neural Magic
 * Adapted from https://github.com/Vahe1994/AQLM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <cstdlib>

namespace vllm {
namespace aqlm {

void Code1x16MatVec(
    const sycl::int4* __restrict__ A, const sycl::int4* __restrict__ B,
    sycl::int4* __restrict__ C, const sycl::int4* __restrict__ codebook,
    const int prob_m, const int prob_k,
    const sycl::int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                        // codebook, at most 3 long.
    const int codebook_stride, const sycl::nd_item<3>& item_ct1,
    sycl::int4* sh_b  // as int4.
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x();
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + item_ct1.get_local_id(2) % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - item_ct1.get_local_id(2) % 32;

  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    /*
    DPCT1118:65: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = item_ct1.get_local_id(2); i < 32 * 8;
         i += item_ct1.get_local_range(2)) {
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    /*
    DPCT1118:66: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (item_ct1.get_local_id(2) % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        uint32_t dec[4];
        // We bypass the L1 cache to avoid massive amounts of memory streaming
        // that doesn't actually help us; this brings > 2x speedup.
        /*
        DPCT1053:67: Migration of device assembly code is not supported.
        */
        asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                     : "l"((void*)&codebook[enc[i]]));
        sycl::half2* a = reinterpret_cast<sycl::half2*>(&dec);
        sycl::half2* b = reinterpret_cast<sycl::half2*>(&sh_b[b_sh_rd]);
        sycl::half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++) res2 = sycl::fma(a[j], b[j], res2);
        res += sycl::vec<sycl::half, 1>(res2.x())
                   .convert<float, sycl::rounding_mode::automatic>()[0] +
               sycl::vec<sycl::half, 1>(res2.y())
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    /*
    DPCT1096:417: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int i = 16; i > 0; i /= 2) res +=
        dpct::shift_sub_group_left(item_ct1.get_sub_group(), res, i);
    if (item_ct1.get_local_id(2) % 32 == 0)
      reinterpret_cast<sycl::half*>(C)[c_gl_wr] =
          sycl::vec<float, 1>(res)
              .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

/*
DPCT1110:68: The total declared local variable size in device function
Code2x8MatVec exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void Code2x8MatVec(
    const sycl::int4* __restrict__ A, const sycl::int4* __restrict__ B,
    sycl::int4* __restrict__ C, const sycl::int4* __restrict__ codebook,
    int prob_m, int prob_k,
    const sycl::int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                        // codebook, at most 3 long.
    const int codebook_stride, const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local  // as int4.

) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x();
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + item_ct1.get_local_id(2) % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - item_ct1.get_local_id(2) % 32;
  int lane = item_ct1.get_local_id(2) % 8;

  auto sh = (sycl::int4*)dpct_local;
  sycl::int4* sh_b = sh;
  sycl::int4* sh_code = sh_b + 32 * 9;
  sycl::int4* sh_code0 = sh_code;
  sycl::int4* sh_code1 = sh_code + 256 * 8;

  for (int i = item_ct1.get_local_id(2); i < 2 * 256;
       i += item_ct1.get_local_range(2)) {
    sycl::int4 dec = codebook[i];
#pragma unroll
    for (int j = 0; j < 8; j++) sh_code[8 * i + (j + lane) % 8] = dec;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  float res = 0;

  int iters = (prob_k / 8 + 8 * 32 - 1) / (8 * 32);
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    /*
    DPCT1118:69: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int i = item_ct1.get_local_id(2); i < 32 * 8;
         i += item_ct1.get_local_range(2)) {
      if (b_gl_rd + i < prob_k / 8) sh_b[9 * (i / 8) + i % 8] = B[b_gl_rd + i];
    }
    /*
    DPCT1118:70: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    b_gl_rd += 32 * 8;

    int b_sh_rd = 9 * (item_ct1.get_local_id(2) % 32);
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        sycl::half2* a0 = reinterpret_cast<sycl::half2*>(
            &sh_code0[8 * enc[2 * i + 0] + lane]);
        sycl::half2* a1 = reinterpret_cast<sycl::half2*>(
            &sh_code1[8 * enc[2 * i + 1] + lane]);
        sycl::half2* b = reinterpret_cast<sycl::half2*>(&sh_b[b_sh_rd]);
        sycl::half2 res2 = {};
#pragma unroll
        for (int j = 0; j < 4; j++)
          res2 = sycl::fma(a0[j] + a1[j], b[j], res2);
        res += sycl::vec<sycl::half, 1>(res2.x())
                   .convert<float, sycl::rounding_mode::automatic>()[0] +
               sycl::vec<sycl::half, 1>(res2.y())
                   .convert<float, sycl::rounding_mode::automatic>()[0];
        b_sh_rd++;
      }
      a_gl_rd += 32;
    }
  }

  if (pred) {
#pragma unroll
    /*
    DPCT1096:418: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::shift_sub_group_left" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    for (int i = 16; i > 0; i /= 2) res +=
        dpct::shift_sub_group_left(item_ct1.get_sub_group(), res, i);
    if (item_ct1.get_local_id(2) % 32 == 0)
      reinterpret_cast<sycl::half*>(C)[c_gl_wr] =
          sycl::vec<float, 1>(res)
              .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void Code2x8MatVec_wrapper(const sycl::int4* __restrict A,
                           const sycl::int4* __restrict B,
                           sycl::int4* __restrict C,
                           const sycl::int4* __restrict codebook, int prob_m,
                           int prob_k, const sycl::int4 codebook_a_sizes,
                           const int codebook_stride) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(localMemSize), cgh);

    cgh.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
          Code2x8MatVec(
              A, B, C, codebook, prob_m, prob_k, codebook_a_sizes,
              codebook_stride, item_ct1,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}

void Code1x16Dequant(
    const sycl::int4* __restrict__ A, sycl::int4* __restrict__ C,
    const sycl::int4* __restrict__ codebook, int prob_m, int prob_k,
    const sycl::int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                        // codebook, at most 3 long, sums to m.
    const int codebook_stride, const sycl::nd_item<3>& item_ct1  // as int4
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x();
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  a_gl_rd = a_gl_stride * a_gl_rd + item_ct1.get_local_id(2) % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - item_ct1.get_local_id(2) % 32;

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (item_ct1.get_local_id(2) % 32) * 8;

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint16_t* enc = reinterpret_cast<const uint16_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        sycl::int4 chunk;
        auto dec = reinterpret_cast<uint32_t*>(&chunk);
        // We bypass the L1 cache to avoid massive amounts of memory streaming
        // that doesn't actually help us; this brings > 2x speedup.
        /*
        DPCT1053:71: Migration of device assembly code is not supported.
        */
        asm volatile("ld.cg.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(dec[0]), "=r"(dec[1]), "=r"(dec[2]), "=r"(dec[3])
                     : "l"((void*)&codebook[enc[i]]));

        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

/*
DPCT1110:72: The total declared local variable size in device function
Code2x8Dequant exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void Code2x8Dequant(
    const sycl::int4* __restrict__ A, sycl::int4* __restrict__ C,
    const sycl::int4* __restrict__ codebook, int prob_m, int prob_k,
    const sycl::int4
        codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at
                           // most 3 long, corresponds to cols.
    const int codebook_stride, const sycl::nd_item<3>& item_ct1,
    uint8_t* dpct_local  // as int4
) {
  int a_gl_stride = prob_k / 8 / 8;
  int a_gl_rd = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  bool pred = a_gl_rd < prob_m;

  if (pred) {
    // advance to the correct codebook, this easy because we only multiply one
    // column of the codebook.
    auto codebook_size = &codebook_a_sizes.x();
    while (a_gl_rd >= *codebook_size) {
      codebook += codebook_stride;
      ++codebook_size;
    }
  }

  a_gl_rd = a_gl_stride * a_gl_rd + item_ct1.get_local_id(2) % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - item_ct1.get_local_id(2) % 32;
  int lane = item_ct1.get_local_id(2) % 8;

  int c_gl_stride = prob_k / 8;
  int c_gl_wr = (item_ct1.get_local_range(2) / 32) * item_ct1.get_group(2) +
                (item_ct1.get_local_id(2) / 32);
  c_gl_wr = c_gl_stride * c_gl_wr + (item_ct1.get_local_id(2) % 32) * 8;

  auto sh = (sycl::int4*)dpct_local;
  sycl::int4* sh_code = sh;
  sycl::int4* sh_code0 = sh_code;
  sycl::int4* sh_code1 = sh_code + 256 * 8;

  for (int i = item_ct1.get_local_id(2); i < 2 * 256;
       i += item_ct1.get_local_range(2)) {
    sycl::int4 dec = codebook[i];
#pragma unroll
    for (int j = 0; j < 8; j++) sh_code[8 * i + (j + lane) % 8] = dec;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  int iters = (prob_k / 8 - 1) / (8 * 32) + 1;
  while (iters--) {
    if (pred && a_gl_rd < a_gl_end) {
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        sycl::int4 chunk;
        sycl::half2* a0 = reinterpret_cast<sycl::half2*>(
            &sh_code0[8 * enc[2 * i + 0] + lane]);
        sycl::half2* a1 = reinterpret_cast<sycl::half2*>(
            &sh_code1[8 * enc[2 * i + 1] + lane]);
#pragma unroll
        for (int j = 0; j < 4; j++)
          reinterpret_cast<sycl::half2*>(&chunk)[j] = a0[j] + a1[j];
        C[a_gl_rd * 8 + i] = chunk;
      }
    }
    a_gl_rd += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void Code2x8Dequant_wrapper(const sycl::int4* __restrict A,
                            sycl::int4* __restrict C,
                            const sycl::int4* __restrict codebook, int prob_m,
                            int prob_k, const sycl::int4 codebook_a_sizes,
                            const int codebook_stride) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(localMemSize), cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      Code2x8Dequant(
          A, C, codebook, prob_m, prob_k, codebook_a_sizes, codebook_stride,
          item_ct1,
          dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
              .get());
    });
  });
}

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

const int THREAD_M = 16;

void code1x16_matvec_cuda(const void* __restrict__ A,
                          const void* __restrict__ B, void* __restrict__ C,
                          const void* __restrict__ codebook, int prob_m,
                          int prob_k, const sycl::int4 codebook_a_sizes,
                          const int codebook_stride) {
  int sms;
  sms = dpct::get_device(0).get_max_compute_units();
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream().stream();
  /*
  DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<sycl::int4, 1> sh_b_acc_ct1(sycl::range<1>(32 * 9),
                                                       cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads),
              sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            Code1x16MatVec(
                (const sycl::int4*)A, (const sycl::int4*)B, (sycl::int4*)C,
                (const sycl::int4*)codebook, prob_m, prob_k, codebook_a_sizes,
                codebook_stride, item_ct1,
                sh_b_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
}

void code2x8_matvec_cuda(const void* __restrict__ A, const void* __restrict__ B,
                         void* __restrict__ C,
                         const void* __restrict__ codebook, int prob_m,
                         int prob_k, const sycl::int4 codebook_a_sizes,
                         const int codebook_stride) {
  int sms;
  sms = dpct::get_device(0).get_max_compute_units();
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  cudaFuncSetAttribute(Code2x8MatVec_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream().stream();
  /*
  DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
          sycl::range<1>(shared), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads),
              sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            Code2x8MatVec(
                (const sycl::int4*)A, (const sycl::int4*)B, (sycl::int4*)C,
                (const sycl::int4*)codebook, prob_m, prob_k, codebook_a_sizes,
                codebook_stride, item_ct1,
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
}

void code1x16_dequant_cuda(
    const void* __restrict__ A, void* __restrict__ C,
    const void* __restrict__ codebook, int prob_m, int prob_k,
    const sycl::int4 codebook_a_sizes,  // cumulative sizes of A spanning each
                                        // codebook, at most 3 long.
    const int codebook_stride           // as int4.
) {
  int sms;
  sms = dpct::get_device(0).get_max_compute_units();
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream().stream();
  /*
  DPCT1049:75: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         Code1x16Dequant((const sycl::int4*)A, (sycl::int4*)C,
                                         (const sycl::int4*)codebook, prob_m,
                                         prob_k, codebook_a_sizes,
                                         codebook_stride, item_ct1);
                       });
}

// Dequantizes the code and codebook into weights.
void code2x8_dequant_cuda(
    const void* __restrict__ A, void* __restrict__ C,
    const void* __restrict__ codebook, int prob_m, int prob_k,
    const sycl::int4
        codebook_a_sizes,  // cumulative sizes of A spanning each codebook, at
                           // most 3 long, corresponds to cols.
    const int codebook_stride  // as int4
) {
  int sms;
  sms = dpct::get_device(0).get_max_compute_units();
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * (2 * 256 * 8 + 32 * 9);
  dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream().stream();

  cudaFuncSetAttribute(Code2x8Dequant_wrapper,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  /*
  DPCT1049:76: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
          sycl::range<1>(shared), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threads),
              sycl::range<3>(1, 1, threads)),
          [=](sycl::nd_item<3> item_ct1) {
            Code2x8Dequant(
                (const sycl::int4*)A, (sycl::int4*)C,
                (const sycl::int4*)codebook, prob_m, prob_k, codebook_a_sizes,
                codebook_stride, item_ct1,
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
}

int codebook_stride(const torch::Tensor& codebooks) {
  return codebooks.stride(0) * codebooks.element_size() / sizeof(sycl::int4);
}

void code1x16_matvec(
    const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
    const torch::Tensor& codebook,
    const sycl::int4 codebook_a_sizes  // cumulative sizes of A spanning each
                                       // codebook, at most 3 long.
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  code1x16_matvec_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(),
                       codebook.data_ptr(), prob_m, prob_k, codebook_a_sizes,
                       codebook_stride(codebook));
}

torch::Tensor code1x16_matmat(const torch::Tensor& input,
                              const torch::Tensor& codes,
                              const torch::Tensor& codebooks,
                              const torch::Tensor& scales,
                              const sycl::int4 codebook_a_sizes,
                              const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty(
      {flat_input.size(0), out_features},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code1x16_matvec(codes.squeeze(2), input_vec, output_vec, codebooks,
                    codebook_a_sizes);
  }
  flat_output *= scales.flatten().unsqueeze(0);

  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

void code2x8_matvec(const torch::Tensor& A, const torch::Tensor& B,
                    torch::Tensor& C, const torch::Tensor& codebook,
                    const sycl::int4 codebook_a_sizes) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);
  code2x8_matvec_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(),
                      codebook.data_ptr(), prob_m, prob_k, codebook_a_sizes,
                      2 * codebook_stride(codebook));
}

torch::Tensor code2x8_matmat(const torch::Tensor& input,
                             const torch::Tensor& codes,
                             const torch::Tensor& codebooks,
                             const torch::Tensor& scales,
                             const sycl::int4 codebook_a_sizes,
                             const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty(
      {flat_input.size(0), out_features},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    code2x8_matvec(codes.squeeze(2), input_vec, output_vec, codebooks,
                   codebook_a_sizes);
  }
  flat_output *= scales.flatten().unsqueeze(0);
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  auto output = flat_output.reshape(output_sizes);
  return output;
}

// Accumulate the partition sizes.
sycl::int4 accumulate_sizes(
    const std::vector<int64_t>& codebook_partition_sizes) {
  sycl::int4 cumulative_sizes;
  auto cumulative_size = &cumulative_sizes.x();
  size_t i = 0;
  int last = 0;
  assert(codebook_partition_sizes.size() <= 4);
  for (; i < codebook_partition_sizes.size(); ++i, ++cumulative_size) {
    *cumulative_size = codebook_partition_sizes[i] + last;
    last = *cumulative_size;
  }
  // fill in the rest with unreachable.
  for (; i < 4; ++i, ++cumulative_size) {
    *cumulative_size = last * 10;
  }
  return cumulative_sizes;
}

}  // namespace aqlm
}  // namespace vllm

torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const std::vector<int64_t>& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias) {
  sycl::int4 cumulative_sizes =
      vllm::aqlm::accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size();
  int const entries = codebooks.size(1);

  if (nbooks == 1 && entries == (1 << 16)) {
    return vllm::aqlm::code1x16_matmat(input, codes, codebooks, scales,
                                       cumulative_sizes, bias);
  }
  if (nbooks == 2 && entries == (1 << 8)) {
    return vllm::aqlm::code2x8_matmat(input, codes, codebooks, scales,
                                      cumulative_sizes, bias);
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries,
              " entries is not currently supported.")
  return {};
}

torch::Tensor aqlm_dequant(
    const torch::Tensor& codes, const torch::Tensor& codebooks,
    const std::vector<int64_t>& codebook_partition_sizes) {
  sycl::int4 cumulative_sizes =
      vllm::aqlm::accumulate_sizes(codebook_partition_sizes);

  int const nbooks = codebooks.size(0) / codebook_partition_sizes.size();
  int const entries = codebooks.size(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(codes));
  int rows = codes.size(1);
  int cols = codes.size(0);

  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0);

  assert(out_features == std::accumulate(codebook_partition_sizes.begin(),
                                         codebook_partition_sizes.end(), 0));

  auto weights = torch::empty({out_features, in_features},
                              torch::TensorOptions()
                                  .dtype(codebooks.dtype())
                                  .device(codebooks.device()));

  if (nbooks == 1 && entries == (1 << 16)) {
    vllm::aqlm::code1x16_dequant_cuda(codes.data_ptr(), weights.data_ptr(),
                                      codebooks.data_ptr(), out_features,
                                      in_features, cumulative_sizes,
                                      vllm::aqlm::codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower
    // and not consistent with gemv implementation.) weights *=
    // scales.index({"...", 0, 0});

    return weights;
  }

  if (nbooks == 2 && entries == (1 << 8)) {
    vllm::aqlm::code2x8_dequant_cuda(codes.data_ptr(), weights.data_ptr(),
                                     codebooks.data_ptr(), out_features,
                                     in_features, cumulative_sizes,
                                     vllm::aqlm::codebook_stride(codebooks));

    // if you wanted to flip to scaling the weights, (though it's 30%-ish slower
    // and not consistent with gemv implementation) weights *=
    // scales.index({"...", 0, 0});

    return weights;
  }

  TORCH_CHECK(false, "AQLM with ", nbooks, " codebooks and ", entries,
              " entries is not currently supported.")
  return {};
}
