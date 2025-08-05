/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "compat.dp.hpp"
#include "matrix_view.dp.hpp"
#include "qdq_2.dp.hpp"
#include "qdq_3.dp.hpp"
#include "qdq_4.dp.hpp"
#include "qdq_8.dp.hpp"
#include <cmath>

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
#define MAX_Q_GEMM_ROWS_8BIT 24
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM)
  #include <hipblas/hipblas.h>
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transA,
    hipblasOperation_t transB, int m, int n, int k, const half* alpha,
    const half* AP, int lda, const half* BP, int ldb, const half* beta,
    half* CP, int ldc) {
  return hipblasHgemm(handle, transA, transB, m, n, k,
                      reinterpret_cast<const hipblasHalf*>(alpha),
                      reinterpret_cast<const hipblasHalf*>(AP), lda,
                      reinterpret_cast<const hipblasHalf*>(BP), ldb,
                      reinterpret_cast<const hipblasHalf*>(beta),
                      reinterpret_cast<hipblasHalf*>(CP), ldc);
}
  #define hipblasHgemm __compat_hipblasHgemm

  // Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
  #define rocblas_operation_none HIPBLAS_OP_N
  #define rocblas_hgemm __compat_hipblasHgemm
#endif

__dpct_inline__ sycl::half2 dot22_8(sycl::half2 (&dq)[4],
                                    const sycl::half* a_ptr,
                                    const sycl::half2 g_result) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  return result + g_result;
}

__dpct_inline__ float dot22_8_f(sycl::half2 (&dq)[4], const sycl::half* a_ptr) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  return sycl::vec<sycl::half, 1>(result[0])
             .convert<float, sycl::rounding_mode::automatic>()[0] +
         sycl::vec<sycl::half, 1>(result[1])
             .convert<float, sycl::rounding_mode::automatic>()[0];
}

__dpct_inline__ sycl::half2 dot22_8(sycl::half2 (&dq)[4],
                                    const sycl::half* a_ptr,
                                    const sycl::half2 g_result,
                                    const sycl::half qs_h) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  return sycl::fma(result, sycl::half2(qs_h, qs_h), g_result);
}

__dpct_inline__ sycl::half2 dot22_16(sycl::half2 (&dq)[8],
                                     const sycl::half* a_ptr,
                                     const sycl::half2 g_result,
                                     const sycl::half qs_h) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  return sycl::fma(result, sycl::half2(qs_h, qs_h), g_result);
}

__dpct_inline__ sycl::half2 dot22_32(sycl::half2 (&dq)[16],
                                     const sycl::half* a_ptr,
                                     const sycl::half2 g_result,
                                     const sycl::half qs_h) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = sycl::fma(dq[i], *a2_ptr++, result);
  return sycl::fma(result, sycl::half2(qs_h, qs_h), g_result);
}

__dpct_inline__ float dot22_8_f(sycl::half2 (&dq)[4], const sycl::half* a_ptr,
                                const float g_result, const float qs_f) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  float result_f = sycl::vec<sycl::half, 1>(result[0])
                       .convert<float, sycl::rounding_mode::automatic>()[0] +
                   sycl::vec<sycl::half, 1>(result[1])
                       .convert<float, sycl::rounding_mode::automatic>()[0];
  return sycl::fma(result_f, (float)qs_f, (float)g_result);
}

__dpct_inline__ float dot22_16_f(sycl::half2 (&dq)[8], const sycl::half* a_ptr,
                                 const float g_result, const float qs_f) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  float result_f = sycl::vec<sycl::half, 1>(result[0])
                       .convert<float, sycl::rounding_mode::automatic>()[0] +
                   sycl::vec<sycl::half, 1>(result[1])
                       .convert<float, sycl::rounding_mode::automatic>()[0];
  return sycl::fma(result_f, (float)qs_f, (float)g_result);
}

__dpct_inline__ float dot22_32_f(sycl::half2 (&dq)[16], const sycl::half* a_ptr,
                                 const float g_result, const float qs_f) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = sycl::fma(dq[i], *a2_ptr++, result);
  float result_f = sycl::vec<sycl::half, 1>(result[0])
                       .convert<float, sycl::rounding_mode::automatic>()[0] +
                   sycl::vec<sycl::half, 1>(result[1])
                       .convert<float, sycl::rounding_mode::automatic>()[0];
  return sycl::fma(result_f, (float)qs_f, (float)g_result);
}

__dpct_inline__ sycl::half dot22_8_h(sycl::half2 (&dq)[4],
                                     const sycl::half* a_ptr,
                                     const sycl::half g_result,
                                     const sycl::half qs_h) {
  // Use FP32 accumulator to avoid potential overflow since unscaled weights are
  // in the range -128..127

  float result = {};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    sycl::half2 w01 = dq[i];
    float w0 = w01[0];
    float w1 = w01[1];
    float x0 = sycl::vec<sycl::half, 1>(*a_ptr++)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
    float x1 = sycl::vec<sycl::half, 1>(*a_ptr++)
                   .convert<float, sycl::rounding_mode::automatic>()[0];
    result = sycl::fma(w0, x0, result);
    result = sycl::fma(w1, x1, result);
  }
  float qs = sycl::vec<sycl::half, 1>(qs_h)
                 .convert<float, sycl::rounding_mode::automatic>()[0];
  result *= qs;
  sycl::half result_h = sycl::vec<float, 1>(result)
                            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  return result_h + g_result;
}

__dpct_inline__ sycl::half dot22_16_h(sycl::half2 (&dq)[8],
                                      const sycl::half* a_ptr,
                                      const sycl::half g_result,
                                      const sycl::half qs_h) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = sycl::fma(dq[i], *a2_ptr++, result);
  sycl::half result_h = result[0] + result[1];
  return sycl::fma(result_h, qs_h, g_result);
}

__dpct_inline__ sycl::half dot22_32_h(sycl::half2 (&dq)[16],
                                      const sycl::half* a_ptr,
                                      const sycl::half g_result,
                                      const sycl::half qs_h) {
  sycl::half2 result = {};
  const sycl::half2* a2_ptr = (const sycl::half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = sycl::fma(dq[i], *a2_ptr++, result);
  sycl::half result_h = result[0] + result[1];
  return sycl::fma(result_h, qs_h, g_result);
}

typedef void (*fp_gemm_half_q_half_gptq_kernel)(
    const sycl::half*, const uint32_t*, const uint32_t*, const sycl::half*,
    sycl::half*, const int, const int, const int, const int, const int*);

template <bool first_block, int m_count>
/*
DPCT1110:171: The total declared local variable size in device function
gemm_half_q_half_gptq_4bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_gptq_4bit_kernel(
    const sycl::half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, sycl::half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm, const sycl::nd_item<3>& item_ct1,
    sycl::half block_a[m_count][128 /*BLOCK_KN_SIZE*/]) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = item_ct1.get_local_id(2);

  // Block
  auto offset_n = item_ct1.get_group(2) * BLOCK_KN_SIZE * 4;
  auto offset_m = item_ct1.get_group(1) * m_count;
  auto offset_k = item_ct1.get_group(0) * BLOCK_KN_SIZE;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const sycl::half* a_ptr = a_.item_ptr(offset_m + m, 0);
      sycl::half* block_a_ptr = block_a[m];

      sycl::half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const sycl::half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  float scales[4];
  sycl::half2 z1z16[4][2];
  sycl::half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_f(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

  // Column result
  float block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_f(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      const sycl::int4* b_ptr4 = (sycl::int4*)b_ptr;
      sycl::int4 load_int4 = *b_ptr4;

      sycl::half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x(), dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y(), dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z(), dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w(), dq[3], z1z16[3], y1y16[3], size_n,
                          false);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0],
                            block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1],
                            block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2],
                            block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3],
                            block_c[m][3]);
      }

      b_ptr += size_n;
      a_ptr += 8;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    sycl::half2* out = (sycl::half2*)c_.item_ptr(offset_m + m, n);
    sycl::half2 result01 =
        sycl::half2(sycl::vec<float, 1>(block_c[m][0])
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                    sycl::vec<float, 1>(block_c[m][1])
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    sycl::half2 result23 =
        sycl::half2(sycl::vec<float, 1>(block_c[m][2])
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                    sycl::vec<float, 1>(block_c[m][3])
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out, result01);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out + 1, result23);
  }
}

template <bool first_block, int m_count>
/*
DPCT1110:172: The total declared local variable size in device function
gemm_half_q_half_gptq_2bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_gptq_2bit_kernel(
    const sycl::half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, sycl::half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm, const sycl::nd_item<3>& item_ct1,
    sycl::half block_a[m_count][128 /*BLOCK_KN_SIZE*/]) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = item_ct1.get_local_id(2);

  // Block
  auto offset_n = item_ct1.get_group(2) * BLOCK_KN_SIZE * 4;
  auto offset_m = item_ct1.get_group(1) * m_count;
  auto offset_k = item_ct1.get_group(0) * BLOCK_KN_SIZE;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const sycl::half* a_ptr = a_.item_ptr(offset_m + m, 0);
      sycl::half* block_a_ptr = block_a[m];

      sycl::half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const sycl::half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  sycl::half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  sycl::half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      const sycl::int4* b_ptr4 = (sycl::int4*)b_ptr;
      sycl::int4 load_int4 = *b_ptr4;

      sycl::half2 dq[4][8];
      dequant_2bit_16(load_int4.x(), dq[0], size_n, zeros[0] + 1);
      dequant_2bit_16(load_int4.y(), dq[1], size_n, zeros[1] + 1);
      dequant_2bit_16(load_int4.z(), dq[2], size_n, zeros[2] + 1);
      dequant_2bit_16(load_int4.w(), dq[3], size_n, zeros[3] + 1);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }

      b_ptr += size_n;
      a_ptr += 16;
    }

    k += 16;
  }

  for (int m = 0; m < m_count; m++) {
    sycl::half2* out = (sycl::half2*)c_.item_ptr(offset_m + m, n);
    sycl::half2 result01 = sycl::half2(block_c[m][0], block_c[m][1]);
    sycl::half2 result23 = sycl::half2(block_c[m][2], block_c[m][3]);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out, result01);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out + 1, result23);
  }
}

template <bool first_block, int m_count>
/*
DPCT1110:173: The total declared local variable size in device function
gemm_half_q_half_gptq_3bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_gptq_3bit_kernel(
    const sycl::half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, sycl::half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm, const sycl::nd_item<3>& item_ct1,
    sycl::half block_a[m_count][128 /*BLOCK_KN_SIZE*/]) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = item_ct1.get_local_id(2);

  // Block
  auto offset_n = item_ct1.get_group(2) * BLOCK_KN_SIZE * 4;
  auto offset_m = item_ct1.get_group(1) * m_count;
  auto offset_k = item_ct1.get_group(0) * BLOCK_KN_SIZE;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const sycl::half* a_ptr = a_.item_ptr(offset_m + m, 0);
      sycl::half* block_a_ptr = block_a[m];

      sycl::half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const sycl::half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  sycl::half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  sycl::half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      sycl::int4 load_int4[3];
      load_int4[0] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;

      sycl::half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x(), load_int4[1].x(), load_int4[2].x(),
                      dq[0], size_n, zeros[0] + 1);
      dequant_3bit_32(load_int4[0].y(), load_int4[1].y(), load_int4[2].y(),
                      dq[1], size_n, zeros[1] + 1);
      dequant_3bit_32(load_int4[0].z(), load_int4[1].z(), load_int4[2].z(),
                      dq[2], size_n, zeros[2] + 1);
      dequant_3bit_32(load_int4[0].w(), load_int4[1].w(), load_int4[2].w(),
                      dq[3], size_n, zeros[3] + 1);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 32;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    sycl::half2* out = (sycl::half2*)c_.item_ptr(offset_m + m, n);
    sycl::half2 result01 = sycl::half2(block_c[m][0], block_c[m][1]);
    sycl::half2 result23 = sycl::half2(block_c[m][2], block_c[m][3]);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out, result01);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out + 1, result23);
  }
}

template <bool first_block, int m_count>
/*
DPCT1110:174: The total declared local variable size in device function
gemm_half_q_half_gptq_8bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_gptq_8bit_kernel(
    const sycl::half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, sycl::half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm, const sycl::nd_item<3>& item_ct1,
    sycl::half block_a[m_count][128 /*BLOCK_KN_SIZE*/]) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = item_ct1.get_local_id(2);

  // Block
  auto offset_n = item_ct1.get_group(2) * BLOCK_KN_SIZE * 4;
  auto offset_m = item_ct1.get_group(1) * m_count;
  auto offset_k = item_ct1.get_group(0) * BLOCK_KN_SIZE;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const sycl::half* a_ptr = a_.item_ptr(offset_m + m, 0);
      sycl::half* block_a_ptr = block_a[m];

      sycl::half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const sycl::half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  sycl::half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  sycl::half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      sycl::int4 load_int4[2];
      load_int4[0] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;

      sycl::half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x(), load_int4[1].x(), dq[0], size_n,
                     zeros[0] + 1);
      dequant_8bit_8(load_int4[0].y(), load_int4[1].y(), dq[1], size_n,
                     zeros[1] + 1);
      dequant_8bit_8(load_int4[0].z(), load_int4[1].z(), dq[2], size_n,
                     zeros[2] + 1);
      dequant_8bit_8(load_int4[0].w(), load_int4[1].w(), dq[3], size_n,
                     zeros[3] + 1);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    sycl::half2* out = (sycl::half2*)c_.item_ptr(offset_m + m, n);
    sycl::half2 result01 = sycl::half2(block_c[m][0], block_c[m][1]);
    sycl::half2 result23 = sycl::half2(block_c[m][2], block_c[m][3]);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out, result01);
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out + 1, result23);
  }
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel(
    bool first_block, const int m_count, const int bit) {
#define SELECT_KERNEL(M_COUNT)                                             \
  if (m_count == M_COUNT) {                                                \
    if (bit == 2) return gemm_half_q_half_gptq_2bit_kernel<true, M_COUNT>; \
    if (bit == 3) return gemm_half_q_half_gptq_3bit_kernel<true, M_COUNT>; \
    if (bit == 4) return gemm_half_q_half_gptq_4bit_kernel<true, M_COUNT>; \
    if (bit == 8) return gemm_half_q_half_gptq_8bit_kernel<true, M_COUNT>; \
  }
#if BLOCK_M_SIZE_MAX >= 1
  SELECT_KERNEL(1);
#endif
#if BLOCK_M_SIZE_MAX >= 2
  SELECT_KERNEL(2);
#endif
#if BLOCK_M_SIZE_MAX >= 3
  SELECT_KERNEL(3);
#endif
#if BLOCK_M_SIZE_MAX >= 4
  SELECT_KERNEL(4);
#endif
#if BLOCK_M_SIZE_MAX >= 5
  SELECT_KERNEL(5);
#endif
#if BLOCK_M_SIZE_MAX >= 6
  SELECT_KERNEL(6);
#endif
#if BLOCK_M_SIZE_MAX >= 7
  SELECT_KERNEL(7);
#endif
#if BLOCK_M_SIZE_MAX >= 8
  SELECT_KERNEL(8);
#endif
  return NULL;
}

void gemm_half_q_half_cuda_part(const sycl::half* a, const uint32_t* b_q_weight,
                                const uint32_t* b_gptq_qzeros,
                                const sycl::half* b_gptq_scales,
                                const int* b_q_perm, sycl::half* c, int size_m,
                                int size_n, int size_k, int m_count, int groups,
                                int bit) {
  dpct::dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
  gridDim.y = DIVIDE(size_m, m_count);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  fp_gemm_half_q_half_gptq_kernel kernel =
      pick_gemm_half_q_half_gptq_kernel(true, m_count, bit);

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  dpct::kernel_launcher::launch(kernel, gridDim, blockDim, 0, stream, a,
                                b_q_weight, b_gptq_qzeros, b_gptq_scales, c,
                                size_m, size_n, size_k, groups, b_q_perm);
}

/*
DPCT1110:175: The total declared local variable size in device function
reconstruct_exllama_8bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void reconstruct_exllama_8bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict__ b,
    const sycl::nd_item<3>& item_ct1, int* perm) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * item_ct1.get_group(1);
  auto offset_n = BLOCK_KN_SIZE * item_ct1.get_group(2) * 4;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table

  auto t = item_ct1.get_local_id(2);

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  sycl::half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 4; p++) {
      sycl::int4 load_int4[2];
      load_int4[0] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;

      sycl::half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x(), load_int4[1].x(), dq[0], size_n,
                     zeros[0] + 1);
      dequant_8bit_8(load_int4[0].y(), load_int4[1].y(), dq[1], size_n,
                     zeros[1] + 1);
      dequant_8bit_8(load_int4[0].z(), load_int4[1].z(), dq[2], size_n,
                     zeros[2] + 1);
      dequant_8bit_8(load_int4[0].w(), load_int4[1].w(), dq[3], size_n,
                     zeros[3] + 1);

      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(perm[lk++], n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(perm[lk++], n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(offset_k + lk++, n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(offset_k + lk++, n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      }
    }
    k += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void reconstruct_exllama_8bit_kernel_wrapper(
    const uint32_t* __restrict b_q_weight, const int* __restrict b_q_perm,
    const uint32_t* __restrict b_gptq_qzeros,
    const sycl::half* __restrict b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict b) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:422: 'BLOCK_KN_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<int, 1> perm_acc_ct1(
        sycl::range<1>(128 /*BLOCK_KN_SIZE*/), cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      reconstruct_exllama_8bit_kernel(
          b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, size_k, size_n,
          groups, b, item_ct1,
          perm_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

/*
DPCT1110:176: The total declared local variable size in device function
reconstruct_exllama_4bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void reconstruct_exllama_4bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict__ b,
    const sycl::nd_item<3>& item_ct1, int* perm) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * item_ct1.get_group(1);
  auto offset_n = BLOCK_KN_SIZE * item_ct1.get_group(2) * 4;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table

  auto t = item_ct1.get_local_id(2);

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  sycl::half2 scales[4];
  sycl::half2 z1z16[4][2];
  sycl::half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
    }

    for (int p = 0; p < 4; p++) {
      sycl::half2 dq[4][4];
      const sycl::int4* b_ptr4 = (sycl::int4*)b_ptr;
      sycl::int4 load_int4 = *b_ptr4;

      dequant_4bit_8_gptq(load_int4.x(), dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y(), dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z(), dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w(), dq[3], z1z16[3], y1y16[3], size_n,
                          false);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(perm[lk++], n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(perm[lk++], n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(offset_k + lk++, n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(offset_k + lk++, n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      }
    }
    k += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void reconstruct_exllama_4bit_kernel_wrapper(
    const uint32_t* __restrict b_q_weight, const int* __restrict b_q_perm,
    const uint32_t* __restrict b_gptq_qzeros,
    const sycl::half* __restrict b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict b) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:423: 'BLOCK_KN_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<int, 1> perm_acc_ct1(
        sycl::range<1>(128 /*BLOCK_KN_SIZE*/), cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      reconstruct_exllama_4bit_kernel(
          b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, size_k, size_n,
          groups, b, item_ct1,
          perm_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

/*
DPCT1110:177: The total declared local variable size in device function
reconstruct_exllama_3bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void reconstruct_exllama_3bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict__ b,
    const sycl::nd_item<3>& item_ct1, int* perm) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * item_ct1.get_group(1);
  auto offset_n = BLOCK_KN_SIZE * item_ct1.get_group(2) * 4;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table

  auto t = item_ct1.get_local_id(2);

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  sycl::half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 1; p++) {
      sycl::int4 load_int4[3];
      load_int4[0] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((sycl::int4*)b_ptr);
      b_ptr += size_n;

      sycl::half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x(), load_int4[1].x(), load_int4[2].x(),
                      dq[0], size_n, zeros[0] + 1);
      dequant_3bit_32(load_int4[0].y(), load_int4[1].y(), load_int4[2].y(),
                      dq[1], size_n, zeros[1] + 1);
      dequant_3bit_32(load_int4[0].z(), load_int4[1].z(), load_int4[2].z(),
                      dq[2], size_n, zeros[2] + 1);
      dequant_3bit_32(load_int4[0].w(), load_int4[1].w(), load_int4[2].w(),
                      dq[3], size_n, zeros[3] + 1);

      if (b_q_perm) {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(perm[lk++], n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(perm[lk++], n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      } else {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(offset_k + lk++, n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(offset_k + lk++, n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      }
    }
    k += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void reconstruct_exllama_3bit_kernel_wrapper(
    const uint32_t* __restrict b_q_weight, const int* __restrict b_q_perm,
    const uint32_t* __restrict b_gptq_qzeros,
    const sycl::half* __restrict b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict b) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:424: 'BLOCK_KN_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<int, 1> perm_acc_ct1(
        sycl::range<1>(128 /*BLOCK_KN_SIZE*/), cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      reconstruct_exllama_3bit_kernel(
          b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, size_k, size_n,
          groups, b, item_ct1,
          perm_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

/*
DPCT1110:178: The total declared local variable size in device function
reconstruct_exllama_2bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void reconstruct_exllama_2bit_kernel(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const sycl::half* __restrict__ b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict__ b,
    const sycl::nd_item<3>& item_ct1, int* perm) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * item_ct1.get_group(1);
  auto offset_n = BLOCK_KN_SIZE * item_ct1.get_group(2) * 4;

  int end_k = dpct::min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table

  auto t = item_ct1.get_local_id(2);

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  sycl::half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 2; p++) {
      const sycl::int4* b_ptr4 = (sycl::int4*)b_ptr;
      sycl::int4 load_int4 = *b_ptr4;

      sycl::half2 dq[4][8];
      dequant_2bit_16(load_int4.x(), dq[0], size_n, zeros[0] + 1);
      dequant_2bit_16(load_int4.y(), dq[1], size_n, zeros[1] + 1);
      dequant_2bit_16(load_int4.z(), dq[2], size_n, zeros[2] + 1);
      dequant_2bit_16(load_int4.w(), dq[3], size_n, zeros[3] + 1);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(perm[lk++], n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(perm[lk++], n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      } else {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = scales[v] * dq[v][j];
          b_.set4(offset_k + lk++, n, dq[0][j][0], dq[1][j][0], dq[2][j][0],
                  dq[3][j][0]);
          b_.set4(offset_k + lk++, n, dq[0][j][1], dq[1][j][1], dq[2][j][1],
                  dq[3][j][1]);
        }
      }
    }
    k += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void reconstruct_exllama_2bit_kernel_wrapper(
    const uint32_t* __restrict b_q_weight, const int* __restrict b_q_perm,
    const uint32_t* __restrict b_gptq_qzeros,
    const sycl::half* __restrict b_gptq_scales, const int size_k,
    const int size_n, const int groups, sycl::half* __restrict b) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:425: 'BLOCK_KN_SIZE' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<int, 1> perm_acc_ct1(
        sycl::range<1>(128 /*BLOCK_KN_SIZE*/), cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      reconstruct_exllama_2bit_kernel(
          b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, size_k, size_n,
          groups, b, item_ct1,
          perm_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  });
}

void reconstruct_exllama(const uint32_t* b_q_weight,
                         const uint32_t* b_gptq_qzeros,
                         const sycl::half* b_gptq_scales, const int* b_q_perm,
                         sycl::half* out, int height, int width, int groups,
                         int bit) {
  dpct::dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto reconstruct_exllama_kernel = reconstruct_exllama_4bit_kernel_wrapper;
  if (bit == 2) {
    reconstruct_exllama_kernel = reconstruct_exllama_2bit_kernel_wrapper;
  } else if (bit == 3) {
    reconstruct_exllama_kernel = reconstruct_exllama_3bit_kernel_wrapper;
  } else if (bit == 8) {
    reconstruct_exllama_kernel = reconstruct_exllama_8bit_kernel_wrapper;
  }

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  dpct::kernel_launcher::launch(reconstruct_exllama_kernel, gridDim, blockDim,
                                0, stream, b_q_weight, b_q_perm, b_gptq_qzeros,
                                b_gptq_scales, height, width, groups, out);
}

/*
DPCT1110:179: The total declared local variable size in device function
gemm_half_q_half_alt_4bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_alt_4bit_kernel(
    const sycl::half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    sycl::half* __restrict__ mul, const sycl::half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width, const sycl::nd_item<3>& item_ct1,
    sycl::half2 blockvec[8 /*BLOCK_M_SIZE_MAX*/][64 /*blockwidth2*/],
    sycl::half2 deq2[256][8]) {
  int zero_width = width / 8;
  int vec_height = height * 4;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = item_ct1.get_group(1) * BLOCK_M_SIZE_MAX;
  int b_end = dpct::min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * item_ct1.get_group(0) / 8;
  int h_end = dpct::min(BLOCK_KN_SIZE / 8, height - h) * 4;
  auto w = BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);

  if (item_ct1.get_local_id(2) < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][item_ct1.get_local_id(2)] =
          vec[(m + b) * vec_height + item_ct1.get_group(0) * BLOCK_KN_SIZE / 2 +
              item_ct1.get_local_id(2)];
    }
  }

  auto val = item_ct1.get_local_id(2) / 8;
  auto off = item_ct1.get_local_id(2) % 8;
  for (; val < 256; val += BLOCK_KN_SIZE / 8) {
    deq2[val][off] =
        sycl::half2(sycl::vec<int, 1>(val & 0xF)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                    sycl::vec<int, 1>(val >> 4)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
  }

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] =
        sycl::vec<int, 1>(0).convert<sycl::half, sycl::rounding_mode::rte>()[0];
  }
  /*
  DPCT1065:357: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;
  int z_w = w / 8;
  int z_mod = (w % 8) * 4;
  sycl::half2 res2;
  sycl::half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    sycl::half2 scales_tmp[4];
    sycl::half2 zeros_tmp[4];
    for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      sycl::half scale_f = scales[g * width + w];
      sycl::half scale_f2 = scales[g2 * width + w];
      sycl::half2 scale = sycl::half2(scale_f, scale_f2);
      sycl::half2 zero = sycl::half2(
          scale_f * sycl::vec<int, 1>(
                        -((zeros[g * zero_width + z_w] >> z_mod) & 0xF) - 1)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
          scale_f2 * sycl::vec<int, 1>(
                         -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF) - 1)
                         .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      res2 = sycl::fma(
          sycl::fma(deq2[(tmp >> 0) & 0xff][off], scales_tmp[0], zeros_tmp[0]),
          blockvec[m][k + 0], res2);
      res2 = sycl::fma(
          sycl::fma(deq2[(tmp >> 8) & 0xff][off], scales_tmp[1], zeros_tmp[1]),
          blockvec[m][k + 1], res2);
      res2 = sycl::fma(
          sycl::fma(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]),
          blockvec[m][k + 2], res2);
      res2 = sycl::fma(
          sycl::fma(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]),
          blockvec[m][k + 3], res2);
#ifndef USE_ROCM
      res[m] = res[m] + res2.x() + res2.y();
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 4;
  }
  for (int m = 0; m < b_end; m++) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[(b + m) * width + w], res[m]);
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void gemm_half_q_half_alt_4bit_kernel_wrapper(
    const sycl::half2* __restrict vec, const uint32_t* __restrict mat,
    sycl::half* __restrict mul, const sycl::half* __restrict scales,
    const uint32_t* __restrict zeros, const int* __restrict g_idx, int batch,
    int height, int width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:426: 'BLOCK_M_SIZE_MAX' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    /*
    DPCT1101:427: 'blockwidth2' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<
        sycl::half2[8 /*BLOCK_M_SIZE_MAX*/][64 /*blockwidth2*/], 0>
        blockvec_acc_ct1(cgh);
    sycl::local_accessor<sycl::half2[256][8], 0> deq2_acc_ct1(cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      gemm_half_q_half_alt_4bit_kernel(vec, mat, mul, scales, zeros, g_idx,
                                       batch, height, width, item_ct1,
                                       blockvec_acc_ct1, deq2_acc_ct1);
    });
  });
}

/*
DPCT1110:180: The total declared local variable size in device function
gemm_half_q_half_alt_8bit_kernel exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_half_q_half_alt_8bit_kernel(
    const sycl::half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    sycl::half* __restrict__ mul, const sycl::half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width, const sycl::nd_item<3>& item_ct1,
    sycl::half2 blockvec[8 /*BLOCK_M_SIZE_MAX*/][64 /*blockwidth2*/]) {
  int zero_width = width / 4;
  int vec_height = height * 2;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = item_ct1.get_group(1) * BLOCK_M_SIZE_MAX;
  int b_end = dpct::min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * item_ct1.get_group(0) / 4;
  int h_end = dpct::min(BLOCK_KN_SIZE / 4, height - h) * 2;
  auto w = BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);

  if (item_ct1.get_local_id(2) < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][item_ct1.get_local_id(2)] =
          vec[(m + b) * vec_height + item_ct1.get_group(0) * BLOCK_KN_SIZE / 2 +
              item_ct1.get_local_id(2)];
    }
  }

  if (item_ct1.get_group(0) == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] =
        sycl::vec<int, 1>(0).convert<sycl::half, sycl::rounding_mode::rte>()[0];
  }
  /*
  DPCT1065:358: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  int z_w = w / 4;
  int z_mod = (w % 4) * 8;
  sycl::half2 res2;
  sycl::half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    sycl::half2 scales_tmp[2];
    sycl::half2 zeros_tmp[2];
    for (int tmp_k = 0; tmp_k < 2; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      sycl::half scale_f = scales[g * width + w];
      sycl::half scale_f2 = scales[g2 * width + w];
      sycl::half2 scale = sycl::half2(scale_f, scale_f2);
      sycl::half2 zero = sycl::half2(
          scale_f * sycl::vec<int, 1>(
                        -((zeros[g * zero_width + z_w] >> z_mod) & 0xff) - 1)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0],
          scale_f2 * sycl::vec<int, 1>(
                         -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xff) - 1)
                         .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      sycl::half2 v12 =
          sycl::half2(sycl::vec<int, 1>(tmp & 0xFF)
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                      sycl::vec<int, 1>((tmp >> 8) & 0xFF)
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
      res2 = sycl::fma(sycl::fma(v12, scales_tmp[0], zeros_tmp[0]),
                       blockvec[m][k + 0], res2);
      sycl::half2 v34 =
          sycl::half2(sycl::vec<int, 1>((tmp >> 16) & 0xFF)
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0],
                      sycl::vec<int, 1>((tmp >> 24) & 0xFF)
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
      res2 = sycl::fma(sycl::fma(v34, scales_tmp[1], zeros_tmp[1]),
                       blockvec[m][k + 1], res2);
#ifndef USE_ROCM
      res[m] = res[m] + res2.x() + res2.y();
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 2;
  }
  for (int m = 0; m < b_end; m++) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[(b + m) * width + w], res[m]);
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void gemm_half_q_half_alt_8bit_kernel_wrapper(
    const sycl::half2* __restrict vec, const uint32_t* __restrict mat,
    sycl::half* __restrict mul, const sycl::half* __restrict scales,
    const uint32_t* __restrict zeros, const int* __restrict g_idx, int batch,
    int height, int width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.submit([&](sycl::handler& cgh) {
    /*
    DPCT1101:428: 'BLOCK_M_SIZE_MAX' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    /*
    DPCT1101:429: 'blockwidth2' expression was replaced with a value.
    Modify the code to use the original expression, provided in comments,
    if it is correct.
    */
    sycl::local_accessor<
        sycl::half2[8 /*BLOCK_M_SIZE_MAX*/][64 /*blockwidth2*/], 0>
        blockvec_acc_ct1(cgh);

    cgh.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
      gemm_half_q_half_alt_8bit_kernel(vec, mat, mul, scales, zeros, g_idx,
                                       batch, height, width, item_ct1,
                                       blockvec_acc_ct1);
    });
  });
}

void gemm_half_q_half_alt(const sycl::half* a, const uint32_t* b_q_weight,
                          const uint32_t* b_gptq_qzeros,
                          const sycl::half* b_gptq_scales, const int* b_g_idx,
                          sycl::half* c, int size_m, int size_n, int size_k,
                          int bit) {
  dpct::dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  auto kernel = gemm_half_q_half_alt_4bit_kernel_wrapper;
  if (bit == 8) {
    kernel = gemm_half_q_half_alt_8bit_kernel_wrapper;
  }

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  dpct::kernel_launcher::launch(kernel, gridDim, blockDim, 0, stream,
                                (const sycl::half2*)a, b_q_weight, c,
                                b_gptq_scales, b_gptq_qzeros, b_g_idx, size_m,
                                size_k / 32 * bit, size_n);
}

template <class T, int bit>
void reconstruct_gptq_kernel(const uint32_t* __restrict__ w,
                             const sycl::half* __restrict__ w_scales,
                             const uint32_t* __restrict__ w_zeros,
                             const int* __restrict__ g_idx, const int height,
                             const int width, const int group,
                             sycl::half* __restrict__ out,
                             const sycl::nd_item<3>& item_ct1) {
  // Start of block

  auto column =
      BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  auto row = item_ct1.get_group(1) * 32 / bit;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  T w_zeros_(w_zeros, group, width);

  uint32_t w_read = w[item_ct1.get_group(1) * width + column];
  sycl::half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int s = 0; s < 32; s += bit) {
    int group = g_idx[row + s / bit];
    sycl::half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;
    sycl::half w_item =
        sycl::vec<int, 1>((int)((w_read >> s) & ((1 << bit) - 1)) - w_zero)
            .convert<sycl::half, sycl::rounding_mode::rte>()[0] *
        w_scale;
    *out_ptr = w_item;
    out_ptr += out_.width;
  }
}

void reconstruct_gptq_3bit_kernel(const uint32_t* __restrict__ w,
                                  const sycl::half* __restrict__ w_scales,
                                  const uint32_t* __restrict__ w_zeros,
                                  const int* __restrict__ g_idx,
                                  const int height, const int width,
                                  const int group, sycl::half* __restrict__ out,
                                  const sycl::nd_item<3>& item_ct1) {
  // Start of block
  auto column =
      BLOCK_KN_SIZE * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  auto row = item_ct1.get_group(1) * 32;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  MatrixView_q3_row w_zeros_(w_zeros, group, width);

  uint32_t w1 = w[(item_ct1.get_group(1) * 3) * width + column];
  uint32_t w2 = w[(item_ct1.get_group(1) * 3 + 1) * width + column];
  uint32_t w3 = w[(item_ct1.get_group(1) * 3 + 2) * width + column];
  sycl::half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int i = 0; i < 32; i += 1) {
    int group = g_idx[row + i];
    sycl::half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;
    int w_item;
    if (i == 10) {
      w_item = (w1 >> 30) | ((w2 << 2) & 0x4);
    } else if (i == 21) {
      w_item = (w2 >> 31) | ((w3 << 1) & 0x6);
    } else if (i < 10) {
      w_item = ((w1 >> (i * 3)) & 0x7);
    } else if (i < 21) {
      w_item = ((w2 >> (i * 3 - 32)) & 0x7);
    } else {
      w_item = ((w3 >> (i * 3 - 64)) & 0x7);
    }
    *out_ptr = sycl::vec<int, 1>(w_item - w_zero)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0] *
               w_scale;
    out_ptr += out_.width;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void reconstruct_gptq_3bit_kernel_wrapper(const uint32_t* __restrict w,
                                          const sycl::half* __restrict w_scales,
                                          const uint32_t* __restrict w_zeros,
                                          const int* __restrict g_idx,
                                          const int height, const int width,
                                          const int group,
                                          sycl::half* __restrict out) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    reconstruct_gptq_3bit_kernel(w, w_scales, w_zeros, g_idx, height, width,
                                 group, out, item_ct1);
  });
}

void reconstruct_gptq(const uint32_t* b_q_weight, const uint32_t* b_gptq_qzeros,
                      const sycl::half* b_gptq_scales, const int* b_g_idx,
                      sycl::half* out, int height, int width, int groups,
                      int bit) {
  dpct::dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, 32 / bit);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto kernel = reconstruct_gptq_kernel<MatrixView_q4_row, 4>;
  if (bit == 2) {
    kernel = reconstruct_gptq_kernel<MatrixView_q2_row, 2>;
  } else if (bit == 8) {
    kernel = reconstruct_gptq_kernel<MatrixView_q8_row, 8>;
  } else if (bit == 3) {
    kernel = reconstruct_gptq_3bit_kernel_wrapper;
    gridDim.y = DIVIDE(height, 32);
  }

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  dpct::kernel_launcher::launch(kernel, gridDim, blockDim, 0, stream,
                                b_q_weight, b_gptq_scales, b_gptq_qzeros,
                                b_g_idx, height, width, groups, out);
}

void gemm_half_q_half_cuda(cublasHandle_t cublas_handle, const sycl::half* a,
                           const uint32_t* b_q_weight,
                           const uint32_t* b_gptq_qzeros,
                           const sycl::half* b_gptq_scales, const int* b_g_idx,
                           sycl::half* c, sycl::half* temp_dq, int size_m,
                           int size_n, int size_k, int groups, bool use_exllama,
                           int bit) {
  bool use_reconstruct;
  if (use_exllama) {
    use_reconstruct = ((bit == 8 && size_m > MAX_Q_GEMM_ROWS_8BIT) ||
                       (bit != 8 && size_m > MAX_Q_GEMM_ROWS));
  } else {
    // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
    // we disabled them for now.
    use_reconstruct = (bit < 4 || size_m > MAX_ALT_GEMM_ROWS);
  }
  if (use_reconstruct) {
    // Reconstruct FP16 matrix, then cuBLAS
    if (use_exllama) {
      reconstruct_exllama(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                          temp_dq, size_k, size_n, groups, bit);
    } else {
      reconstruct_gptq(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                       temp_dq, size_k, size_n, groups, bit);
    }

    const sycl::half alpha =
        sycl::vec<float, 1>(1.0f)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    const sycl::half beta =
        sycl::vec<float, 1>(0.0f)
            .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
                &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
  } else if (use_exllama) {
    // Quantized matmul
    int max_chunks = size_m / BLOCK_M_SIZE_MAX;
    int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks) {
      gemm_half_q_half_cuda_part(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                 b_g_idx, c, last_chunk, size_n, size_k,
                                 BLOCK_M_SIZE_MAX, groups, bit);
    }

    if (last_chunk_size) {
      gemm_half_q_half_cuda_part(a + last_chunk * size_k, b_q_weight,
                                 b_gptq_qzeros, b_gptq_scales, b_g_idx,
                                 c + last_chunk * size_n, last_chunk_size,
                                 size_n, size_k, last_chunk_size, groups, bit);
    }
  } else {
    gemm_half_q_half_alt(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                         c, size_m, size_n, size_k, bit);
  }
}

void shuffle_4bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n,
                                    const sycl::nd_item<3> &item_ct1) {
  auto n = item_ct1.get_group(2) * THREADS_X + item_ct1.get_local_id(2);
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_4bit_8(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 8;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void shuffle_4bit_kernel_wrapper(uint32_t* __restrict b_q_weight,
                                 const int size_k, const int size_n) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    shuffle_4bit_kernel(b_q_weight, size_k, size_n, item_ct1);
  });
}

void shuffle_8bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n,
                                    const sycl::nd_item<3> &item_ct1) {
  auto n = item_ct1.get_group(2) * THREADS_X + item_ct1.get_local_id(2);
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_8bit_4(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 4;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void shuffle_8bit_kernel_wrapper(uint32_t* __restrict b_q_weight,
                                 const int size_k, const int size_n) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    shuffle_8bit_kernel(b_q_weight, size_k, size_n, item_ct1);
  });
}

void shuffle_2bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n,
                                    const sycl::nd_item<3> &item_ct1) {
  auto n = item_ct1.get_group(2) * THREADS_X + item_ct1.get_local_id(2);
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_2bit_16(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 16;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void shuffle_2bit_kernel_wrapper(uint32_t* __restrict b_q_weight,
                                 const int size_k, const int size_n) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    shuffle_2bit_kernel(b_q_weight, size_k, size_n, item_ct1);
  });
}

void shuffle_3bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n,
                                    const sycl::nd_item<3> &item_ct1) {
  auto n = item_ct1.get_group(2) * THREADS_X + item_ct1.get_local_id(2);
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_3bit_32(b_ptr, size_n);
    b_ptr += 3 * size_n;
    k += 32;
  }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void shuffle_3bit_kernel_wrapper(uint32_t* __restrict b_q_weight,
                                 const int size_k, const int size_n) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    shuffle_3bit_kernel(b_q_weight, size_k, size_n, item_ct1);
  });
}

void make_sequential_4bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width,
                                            const sycl::nd_item<3> &item_ct1) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  if (w2_column >= w2_stride) return;
  auto w_new2_row = item_ct1.get_group(1);
  int q_perm_idx = w_new2_row << 3;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 3;
    int w2_subrow = source_row & 0x07;
    int w2_row_shift = w2_subrow << 2;
    int wnew2_row_shift = i << 2;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000f0000000f;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void make_sequential_4bit_kernel_wrapper(const uint32_t* __restrict w,
                                         uint32_t* __restrict w_new,
                                         const int* __restrict q_perm,
                                         const int w_width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    make_sequential_4bit_kernel(w, w_new, q_perm, w_width, item_ct1);
  });
}

void make_sequential_2bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width,
                                            const sycl::nd_item<3> &item_ct1) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  if (w2_column >= w2_stride) return;
  auto w_new2_row = item_ct1.get_group(1);
  int q_perm_idx = w_new2_row << 4;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 16; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 4;
    int w2_subrow = source_row & 0x0f;
    int w2_row_shift = w2_subrow << 1;
    int wnew2_row_shift = i << 1;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000300000003;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void make_sequential_2bit_kernel_wrapper(const uint32_t* __restrict w,
                                         uint32_t* __restrict w_new,
                                         const int* __restrict q_perm,
                                         const int w_width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    make_sequential_2bit_kernel(w, w_new, q_perm, w_width, item_ct1);
  });
}

void make_sequential_3bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width,
                                            const sycl::nd_item<3> &item_ct1) {
  auto w_column = THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  if (w_column >= w_width) return;
  auto w_new_row = item_ct1.get_group(1) * 3;
  auto q_perm_idx = item_ct1.get_group(1) << 5;
  uint32_t dst[3] = {0, 0, 0};

#pragma unroll
  for (int i = 0; i < 32; i++) {
    int source_row = q_perm[q_perm_idx++];
    int z_w = (source_row / 32) * 3;
    int z_mod = source_row % 32;
    int z_bit;

    if (z_mod != 10) {
      if (z_mod != 21) {
        z_bit = z_mod;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }

    uint64_t src;
    if (z_mod == 10) {
      src = (w[z_w * w_width + w_column] >> 30) |
            ((w[(z_w + 1) * w_width + w_column] << 2) & 0x4);
    } else if (z_mod == 21) {
      src = (w[z_w * w_width + w_column] >> 31) |
            ((w[(z_w + 1) * w_width + w_column] << 1) & 0x6);
    } else {
      src = w[z_w * w_width + w_column];
      src >>= z_bit;
      src &= 0x07;
    }

    z_w = 0;
    if (i != 10) {
      if (i != 21) {
        z_bit = i;
        if (z_bit > 21) {
          z_bit *= 3;
          z_bit -= 64;
          z_w += 2;
        } else if (z_bit > 10) {
          z_bit *= 3;
          z_bit -= 32;
          z_w += 1;
        } else {
          z_bit *= 3;
        }
      } else {
        z_w += 1;
      }
    }
    if (i == 10) {
      dst[z_w] |= (src & 0x03) << 30;
      dst[z_w + 1] |= ((src & 0x4) >> 2);
    } else if (i == 21) {
      dst[z_w] |= (src & 0x01) << 31;
      dst[z_w + 1] |= ((src & 0x6) >> 1);
    } else {
      dst[z_w] |= (src << z_bit);
    }
  }
  w_new[w_new_row * w_width + w_column] = dst[0];
  w_new[(w_new_row + 1) * w_width + w_column] = dst[1];
  w_new[(w_new_row + 2) * w_width + w_column] = dst[2];
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void make_sequential_3bit_kernel_wrapper(const uint32_t* __restrict w,
                                         uint32_t* __restrict w_new,
                                         const int* __restrict q_perm,
                                         const int w_width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    make_sequential_3bit_kernel(w, w_new, q_perm, w_width, item_ct1);
  });
}

void make_sequential_8bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width,
                                            const sycl::nd_item<3> &item_ct1) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * item_ct1.get_group(2) + item_ct1.get_local_id(2);
  if (w2_column >= w2_stride) return;
  auto w_new2_row = item_ct1.get_group(1);
  int q_perm_idx = w_new2_row << 2;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 2;
    int w2_subrow = source_row & 0x03;
    int w2_row_shift = w2_subrow << 3;
    int wnew2_row_shift = i << 3;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x000000ff000000ff;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void make_sequential_8bit_kernel_wrapper(const uint32_t* __restrict w,
                                         uint32_t* __restrict w_new,
                                         const int* __restrict q_perm,
                                         const int w_width) {
  sycl::queue queue = *dpct::kernel_launcher::_que;
  unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    make_sequential_8bit_kernel(w, w_new, q_perm, w_width, item_ct1);
  });
}

void shuffle_exllama_weight(uint32_t* q_weight, int* q_perm, int height,
                            int width, int bit) {
  if (q_perm) {
    uint32_t* new_qweight = NULL;
    cudaMalloc(&new_qweight, height / 32 * bit * width * sizeof(uint32_t));

    dpct::dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 32 * bit;

    auto kernel = make_sequential_4bit_kernel_wrapper;
    if (bit == 2) {
      kernel = make_sequential_2bit_kernel_wrapper;
    } else if (bit == 3) {
      kernel = make_sequential_3bit_kernel_wrapper;
      gridDim.y = height / 32;
    } else if (bit == 8) {
      kernel = make_sequential_8bit_kernel_wrapper;
    }
    const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
    dpct::kernel_launcher::launch(kernel, gridDim, blockDim, 0, stream,
                                  q_weight, new_qweight, q_perm, width);
    // Replace qweights
    /*
    DPCT1124:359: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    dpct::get_in_order_queue().memcpy(
        q_weight, new_qweight, height / 32 * bit * width * sizeof(uint32_t));
    // Cleanup
    dpct::get_current_device().queues_wait_and_throw();
    dpct::dpct_free(new_qweight, dpct::get_in_order_queue());
  }
  dpct::dim3 blockDim, gridDim;
  blockDim.x = THREADS_X;
  blockDim.y = 1;
  gridDim.x = DIVIDE(width, THREADS_X);
  gridDim.y = 1;
  auto shuffle_kernel = shuffle_4bit_kernel_wrapper;
  if (bit == 2) {
    shuffle_kernel = shuffle_2bit_kernel_wrapper;
  } else if (bit == 3) {
    shuffle_kernel = shuffle_3bit_kernel_wrapper;
  } else if (bit == 8) {
    shuffle_kernel = shuffle_8bit_kernel_wrapper;
  }
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  dpct::kernel_launcher::launch(shuffle_kernel, gridDim, blockDim, 0, stream,
                                q_weight, height, width);
}

}  // namespace gptq
}  // namespace vllm

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, options);

  vllm::gptq::gemm_half_q_half_cuda(
      at::cuda::getCurrentCUDABlasHandle(), (const half*)a.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_gptq_qzeros.data_ptr(),
      (const half*)b_gptq_scales.data_ptr(),
      b_g_idx.device().is_meta() ? NULL : (const int*)b_g_idx.data_ptr(),
      (half*)c.data_ptr(), (half*)temp_dq.data_ptr(),
      c.size(0),              // m
      c.size(1),              // n
      a.size(1),              // k
      b_gptq_qzeros.size(0),  // group number
      use_exllama, bit);
  return c;
}

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_weight));
  vllm::gptq::shuffle_exllama_weight(
      (uint32_t*)q_weight.data_ptr(),
      q_perm.device().is_meta() || q_perm.numel() == 0
          ? NULL
          : (int*)q_perm.data_ptr(),
      q_weight.size(0) * 32 / bit, q_weight.size(1), bit);
}
