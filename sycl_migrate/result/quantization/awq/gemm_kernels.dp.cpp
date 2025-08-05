/*
Adapted from https://github.com/mit-han-lab/llm-awq
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration}, author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang,
Shang and Dang, Xingyu and Han, Song}, journal={arXiv}, year={2023}
}
 */

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "dequantize.dp.hpp"

namespace vllm {
namespace awq {

template <int N>
/*
DPCT1110:2: The total declared local variable size in device function
gemm_forward_4bit_cuda_m16nXk32 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void gemm_forward_4bit_cuda_m16nXk32(
    int G, int split_k_iters, sycl::half* __restrict__ A, int* __restrict__ B,
    sycl::half* __restrict__ scaling_factors, int* __restrict__ zeros, int M,
    int IC, int OC, sycl::half* __restrict__ C,
    const sycl::nd_item<3>& item_ct1, sycl::half* A_shared,
    sycl::half* B_shared) {
  // Only support matrix n = 64 or 128
  assert(N == 64 || N == 128);
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 750
  assert(false);
#else
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];

  int j_factors1 = ((OC + N - 1) / N);
  int blockIdx_y = item_ct1.get_group(2) % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = item_ct1.get_group(2) / ((M + 16 - 1) / 16 * j_factors1);

  sycl::half A_shared_warp[8];
  sycl::half B_shared_warp[N / 4];
  for (int j_0_4_init = 0; j_0_4_init < N / 32; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / N;
  // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
  bool ld_A_flag =
      (blockIdx_y / j_factors1 * 16 +
       item_ct1.get_local_id(1) * row_stride_warp +
       item_ct1.get_local_id(2) * 8 / 32) < M;  // threadIdx.y is warp_id
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  sycl::half* A_ptr = A +
                      (((int)blockIdx_y) / j_factors1 * 16 +
                       (((int)item_ct1.get_local_id(1)) * row_stride_warp) +
                       ((int)item_ct1.get_local_id(2)) / (32 / 8)) *
                          IC +
                      (((int)item_ct1.get_local_id(2)) % (32 / 8)) * 8;

  int* B_ptr = B + ((int)item_ct1.get_local_id(1)) * (OC / 8) * (256 / N) +
               (((int)item_ct1.get_local_id(2)) / (N / 8)) * (OC / 8) +
               (((int)blockIdx_y) % j_factors1) * (N / 8) +
               (((int)item_ct1.get_local_id(2)) % (N / 8)) * 1;
  // Why * 1 in the above line?

  sycl::half* A_shared_ptr =
      A_shared + ((int)item_ct1.get_local_id(1)) * row_stride_warp * (32 + 8) +
      (((int)item_ct1.get_local_id(2)) / (32 / 8)) * (32 + 8) +
      (((int)item_ct1.get_local_id(2)) % (32 / 8)) * 8;

  sycl::half* B_shared_ptr =
      B_shared + ((int)item_ct1.get_local_id(1)) * (row_stride / 2) * (N + 8) +
      (((int)item_ct1.get_local_id(2)) / (N / 8)) * (N + 8) +
      (((int)item_ct1.get_local_id(2)) % (N / 8)) * 8;

  int* zeros_ptr = zeros + (((int)blockIdx_y) % j_factors1) * (N / 8) +
                   ((int)item_ct1.get_local_id(2)) % (N / 8);

  sycl::half* scaling_factors_ptr =
      scaling_factors + (((int)blockIdx_y) % j_factors1) * N +
      (((int)item_ct1.get_local_id(2)) % (N / 8)) * 8;

  sycl::half* C_ptr =
      C +
      static_cast<long long>(blockIdx_z) * M * OC  // blockIdz.x -> split_k dim
      + (((int)blockIdx_y) % j_factors1) * N +
      ((int)item_ct1.get_local_id(1)) * (N / 2) +
      (((int)item_ct1.get_local_id(2)) % 4) * 2;

  // preload s.f. and zeros
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    /*
    DPCT1118:3: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag) {
      *(sycl::uint4*)(A_shared_ptr) = *(sycl::uint4*)(A_ptr + (k_0_0 * 32));
    } else {
      *(sycl::uint4*)(A_shared_ptr) = sycl::uint4(0, 0, 0, 0);
    }

    // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    sycl::uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    sycl::uint4 B_loaded_scale =
        *(sycl::uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 &&
    threadIdx.y == 0){ printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x,
    B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x,
    B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < N / 16; ++ax0_ax1_fused_0) {
      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus
      // zero -> WB UINT4
      // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) *
      // 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15)
      // * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 *
      // 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) *
      // 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) *
      // 8))); row stride in shared memory: (NWARPS * 32 * 8 / cta_N)
      uint32_t B_loaded =
          *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      sycl::uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);

      // - zero and * scale
      // TODO (Haotian): can save 4 assembly instructions if sormulate as deq =
      // q * scale - zero * scale.
      /*
      DPCT1053:5: Migration of device assembly code is not supported.
      */
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.x())
                   : "r"(B_loaded_fp16.x()), "r"(B_loaded_zero.x()));
      B_loaded_fp16.x() =
          sycl::fma(sycl::vec<uint32_t, 1>(B_loaded_fp16.x())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(B_loaded_scale.x())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
              .template as<sycl::vec<uint32_t, 1>>()
              .x();
      /*
      DPCT1053:6: Migration of device assembly code is not supported.
      */
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.y())
                   : "r"(B_loaded_fp16.y()), "r"(B_loaded_zero.y()));
      B_loaded_fp16.y() =
          sycl::fma(sycl::vec<uint32_t, 1>(B_loaded_fp16.y())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(B_loaded_scale.y())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
              .template as<sycl::vec<uint32_t, 1>>()
              .x();
      /*
      DPCT1053:7: Migration of device assembly code is not supported.
      */
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.z())
                   : "r"(B_loaded_fp16.z()), "r"(B_loaded_zero.z()));
      B_loaded_fp16.z() =
          sycl::fma(sycl::vec<uint32_t, 1>(B_loaded_fp16.z())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(B_loaded_scale.z())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
              .template as<sycl::vec<uint32_t, 1>>()
              .x();
      /*
      DPCT1053:8: Migration of device assembly code is not supported.
      */
      asm volatile("sub.f16x2 %0, %1, %2;\n"
                   : "=r"(B_loaded_fp16.w())
                   : "r"(B_loaded_fp16.w()), "r"(B_loaded_zero.w()));
      B_loaded_fp16.w() =
          sycl::fma(sycl::vec<uint32_t, 1>(B_loaded_fp16.w())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(B_loaded_scale.w())
                        .template as<sycl::half2>(),
                    sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
              .template as<sycl::vec<uint32_t, 1>>()
              .x();
      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 ==
      0 && threadIdx.x == 17 && threadIdx.y == 0){ printf("[x] %X %X %X %X\n",
      B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // write back
      *(sycl::uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (N + 8)) =
          B_loaded_fp16;
    }
    /*
    DPCT1118:4: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
        /*
        DPCT1053:9: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
            "addr; }\n"
            : "=r"(addr)
            : "l"((void*)((&(A_shared[(k_0_1 * 16)])) +
                          (((((int)item_ct1.get_local_id(2)) & 15) * 40) +
                           ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));

        /*
        DPCT1053:10: Migration of device assembly code is not supported.
        */
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned*)(A_shared_warp + 0))[0]),
              "=r"(((unsigned*)(A_shared_warp + 0))[1]),
              "=r"(((unsigned*)(A_shared_warp + 0))[2]),
              "=r"(((unsigned*)(A_shared_warp + 0))[3])
            : "r"(addr));
      }

      for (int ax1_0 = 0; ax1_0 < N / 32; ++ax1_0) {
        {
          unsigned int addr;
          /*
          DPCT1053:11: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, "
              "addr; }\n"
              : "=r"(addr)
              : "l"(
                  (void*)((&(B_shared[(
                              ((k_0_1 * (N * 16 + 128)) +
                               (((int)item_ct1.get_local_id(1)) * (N / 2))) +
                              (ax1_0 * 16))])) +
                          (((((int)item_ct1.get_local_id(2)) & 15) * (N + 8)) +
                           ((((int)item_ct1.get_local_id(2)) >> 4) * 8)))));
          /*
          DPCT1053:12: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];\n"
              : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]),
                "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
        }
      }
      for (int j_0_4 = 0; j_0_4 < N / 32; ++j_0_4) {
  #if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP == 750
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
  #else
        {
          /*
          DPCT1053:13: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]),
                "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[0]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[1]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[2]),
                "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          /*
          DPCT1053:14: Migration of device assembly code is not supported.
          */
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, "
              "%13};\n"
              : "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
              : "r"(((unsigned*)(A_shared_warp + 0))[0]),
                "r"(((unsigned*)(A_shared_warp + 0))[1]),
                "r"(((unsigned*)(A_shared_warp + 0))[2]),
                "r"(((unsigned*)(A_shared_warp + 0))[3]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]),
                "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]),
                "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

  #endif
      }
    }
  }

  // TODO: Shang: Hoist loop invariance.
  for (int ax1_0_1 = 0; ax1_0_1 < (N / 32); ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 +
                       ((int)item_ct1.get_local_id(2)) / 4 +
                       (local_id % 4) / 2 * 8;
      if (row_offset < M) {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 +
          local_id % 2) =
            sycl::vec<float, 1>(C_warp[(ax1_0_1 * 8) + local_id])
                .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
      }
    }
  }
#endif
}

/*
DPCT1110:15: The total declared local variable size in device function
dequantize_weights exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void dequantize_weights(int* __restrict__ B,
                        sycl::half* __restrict__ scaling_factors,
                        int* __restrict__ zeros, sycl::half* __restrict__ C,
                        int G, const sycl::nd_item<3>& item_ct1) {
  static constexpr uint32_t ZERO = 0x0;
  sycl::half B_shared[32 * (128 + 8)];

  sycl::half* B_shared_ptr2 = B_shared;

  int N = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);  // 2
  int col = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2));
  int row = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
            item_ct1.get_local_id(1);
  int index1 = 8 * col + 8 * row * N;
  sycl::half* C_ptr2 = C + index1;

  int index2 = col + row * N;
  int* B_ptr2 = B + index2;

  int index3 = col + (int)(row / G) * N;
  int* zeros_ptr2 = zeros + index3;
  int index4 = 8 * col + (int)(row / G) * N * 8;
  sycl::half* scaling_factors_ptr2 = scaling_factors + index4;

  uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr2);
  sycl::uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
  sycl::uint4 B_loaded_scale = *(sycl::uint4*)(scaling_factors_ptr2);

  uint32_t B_loaded = *(uint32_t*)B_ptr2;
  sycl::uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
  /*
  DPCT1053:16: Migration of device assembly code is not supported.
  */
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.x())
               : "r"(B_loaded_fp16.x()), "r"(B_loaded_zero.x()));
  B_loaded_fp16.x() =
      sycl::fma(
          sycl::vec<uint32_t, 1>(B_loaded_fp16.x()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(B_loaded_scale.x()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
          .template as<sycl::vec<uint32_t, 1>>()
          .x();
  /*
  DPCT1053:17: Migration of device assembly code is not supported.
  */
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.y())
               : "r"(B_loaded_fp16.y()), "r"(B_loaded_zero.y()));
  B_loaded_fp16.y() =
      sycl::fma(
          sycl::vec<uint32_t, 1>(B_loaded_fp16.y()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(B_loaded_scale.y()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
          .template as<sycl::vec<uint32_t, 1>>()
          .x();
  /*
  DPCT1053:18: Migration of device assembly code is not supported.
  */
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.z())
               : "r"(B_loaded_fp16.z()), "r"(B_loaded_zero.z()));
  B_loaded_fp16.z() =
      sycl::fma(
          sycl::vec<uint32_t, 1>(B_loaded_fp16.z()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(B_loaded_scale.z()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
          .template as<sycl::vec<uint32_t, 1>>()
          .x();
  /*
  DPCT1053:19: Migration of device assembly code is not supported.
  */
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(B_loaded_fp16.w())
               : "r"(B_loaded_fp16.w()), "r"(B_loaded_zero.w()));
  B_loaded_fp16.w() =
      sycl::fma(
          sycl::vec<uint32_t, 1>(B_loaded_fp16.w()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(B_loaded_scale.w()).template as<sycl::half2>(),
          sycl::vec<uint32_t, 1>(ZERO).template as<sycl::half2>())
          .template as<sycl::vec<uint32_t, 1>>()
          .x();

  *(sycl::uint4*)B_shared_ptr2 = B_loaded_fp16;

  for (int i = 0; i < 8; ++i) {
    *(C_ptr2 + i) = B_shared[i];
  }
}

}  // namespace awq
}  // namespace vllm

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  int x_thread = thx;
  int y_thread = thy;

  int x_blocks = 1;
  int y_blocks = 1;
  if (thx == 0) {
    x_thread = qout_c;
  }
  if (thy == 0) {
    y_thread = in_c;
  }
  if (thx == 0 && thy == 0) {
    x_thread = 8;
    y_thread = 8;
    x_blocks = (int)(qout_c / 8);
    y_blocks = (int)(in_c / 8);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_scaling_factors));

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

  dpct::dim3 num_blocks(x_blocks, y_blocks);
  dpct::dim3 threads_per_block(x_thread, y_thread);

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  /*
  DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(
        sycl::nd_range<3>(num_blocks * threads_per_block, threads_per_block),
        [=](sycl::nd_item<3> item_ct1) {
          vllm::awq::dequantize_weights(kernel, scaling_factors, zeros,
                                        de_kernel, G, item_ct1);
        });
  }

  return _de_kernel;
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters) {
  int num_in_feats = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
  at::Tensor _out_feats =
      torch::empty({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
  int num_out_feats = _out_feats.size(-2);
  int num_out_channels = _out_feats.size(-1);

  auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
  auto scaling_factors =
      reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
  int group_size = num_in_channels / _scaling_factors.size(0);

  if (num_out_channels % 64 != 0)
    throw std::invalid_argument("OC is not multiple of cta_N = 64");
  if (num_out_channels % 8 != 0)
    throw std::invalid_argument("OC is not multiple of pack_num = 8");
  if (group_size % 32 != 0)
    throw std::invalid_argument("Group size should be a multiple of 32");
  if (num_out_channels % group_size != 0)
    throw std::invalid_argument("OC is not multiple of Group size");

  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  if (num_out_channels % 128 == 0) {
    int j_factors1 = num_out_channels / 128 / 1;
    dpct::dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 *
                          split_k_iters);
    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dpct::dim3 threads_per_block(32, 2);
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> A_shared_acc_ct1(
            sycl::range<1>(16 * (32 + 8)), cgh);
        sycl::local_accessor<sycl::half, 1> B_shared_acc_ct1(
            sycl::range<1>(32 * (128 + 8)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(num_blocks * threads_per_block,
                              threads_per_block),
            [=](sycl::nd_item<3> item_ct1) {
              vllm::awq::gemm_forward_4bit_cuda_m16nXk32<128>(
                  group_size, split_k_iters, in_feats, kernel, scaling_factors,
                  zeros, num_in_feats, num_in_channels, num_out_channels,
                  out_feats, item_ct1,
                  A_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  B_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else if (num_out_channels % 64 == 0) {
    int j_factors1 = num_out_channels / 64 / 1;
    dpct::dim3 num_blocks(1 * (num_out_feats + 16 - 1) / 16 * j_factors1 *
                          split_k_iters);

    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dpct::dim3 threads_per_block(32, 2);
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> A_shared_acc_ct1(
            sycl::range<1>(16 * (32 + 8)), cgh);
        sycl::local_accessor<sycl::half, 1> B_shared_acc_ct1(
            sycl::range<1>(32 * (64 + 8)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(num_blocks * threads_per_block,
                              threads_per_block),
            [=](sycl::nd_item<3> item_ct1) {
              vllm::awq::gemm_forward_4bit_cuda_m16nXk32<64>(
                  group_size, split_k_iters, in_feats, kernel, scaling_factors,
                  zeros, num_in_feats, num_in_channels, num_out_channels,
                  out_feats, item_ct1,
                  A_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  B_shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
  return _out_feats.sum(0);
}
