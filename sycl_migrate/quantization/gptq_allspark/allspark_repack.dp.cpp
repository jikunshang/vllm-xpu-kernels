#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "allspark_utils.dp.hpp"
#include <torch/all.h>
#include "core/registration.h"

namespace allspark {

// Rearrange B to facilitate Ampere Tensor Core load data
// reorder B from (K, N) to (N_32align / 4, K * 4)
// K % 16 == 0, N % 16 == 0, N_32align % 32 == 0
template <typename FType>
/*
DPCT1110:188: The total declared local variable size in device function
rearrange_kn_weight_as_n32k16_order_ldg16_kernel exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
void rearrange_kn_weight_as_n32k16_order_ldg16_kernel(
    const uint8_t* B, const FType* B_scale, const FType* B_zero,
    uint8_t* B_result, FType* B_scale_result, FType* B_zero_result, const int K,
    const int N, const int N_32align, const sycl::nd_item<3>& item_ct1) {
  const auto lane_id = item_ct1.get_local_id(2) % 32;
  const auto warp_id = item_ct1.get_local_id(2) / 32;

  if (item_ct1.get_group(2) != item_ct1.get_group_range(2) - 1) {
    // Load B
    // per block process 64(k) * 128(n) B elements
    // per warp process 16(k) * 128 B elements
    const int src_row_base_idx =
        item_ct1.get_group(2) * 64 + warp_id * 16 + ((lane_id % 8) / 2) * 2;
    const int src_col_idx =
        item_ct1.get_group(1) * 128 + (lane_id / 8) * 32 + (lane_id % 2) * 16;
    uint8_t B_frag[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int src_row_idx = src_row_base_idx + (i / 2) * 8 + (i % 2);
      int src_offset = src_row_idx * N + src_col_idx;
      bool guard = src_row_idx < K && src_col_idx < N;
      ldg128_cg_0(*reinterpret_cast<uint32_t*>(B_frag[i]),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 1),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 2),
                  *(reinterpret_cast<uint32_t*>(B_frag[i]) + 3), B + src_offset,
                  guard);
    }

    // reorder B
    uint8_t B_reorder_frag[8][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int dst_i = j % 8;
        int dst_j = i + (j / 8) * 4;
        B_reorder_frag[dst_i][dst_j] = B_frag[i][j];
      }
    }

    // Store B
    const auto dst_row_base_idx =
        item_ct1.get_group(1) * (128 / 4) + (lane_id / 8) * 8;
    const int dst_col_idx =
        item_ct1.get_group(2) * (64 * 4) + warp_id * 64 + (lane_id % 8) * 8;
    for (int i = 0; i < 8; ++i) {
      int dst_row_idx = dst_row_base_idx + i;
      int dst_offset = dst_row_idx * K * 4 + dst_col_idx;
      bool guard = (dst_row_base_idx < N_32align / 4) && (dst_col_idx < K * 4);
      if (guard) {
        *reinterpret_cast<sycl::int2*>(B_result + dst_offset) =
            *reinterpret_cast<sycl::int2*>(B_reorder_frag[i]);
      }
    }
  } else {
    // Load B_scale and B_zero
    FType b_scale_reg, b_zero_reg;
    auto src_offset = item_ct1.get_group(1) * 128 + item_ct1.get_local_id(2);
    ldg16_cg_0(b_scale_reg, B_scale + src_offset, src_offset < N);
    if (B_zero != nullptr)
      ldg16_cg_0(b_zero_reg, B_zero + src_offset, src_offset < N);
    int dst_offset = item_ct1.get_group(1) * 128 + warp_id * 32 +
                     (lane_id % 8) * 4 + lane_id / 8;
    if (dst_offset < N_32align) {
      B_scale_result[dst_offset] = b_scale_reg;
      if (B_zero != nullptr) B_zero_result[dst_offset] = b_zero_reg;
    }
  }
}

template <typename FType>
void rearrange_kn_weight_as_n32k16_order_ldg16(
    const uint8_t* B, const FType* B_scale, const FType* B_zero,
    uint8_t* B_result, FType* B_scale_result, FType* B_zero_result,
    const int64_t K, const int64_t N, const int64_t N_32align,
    dpct::queue_ptr stream) {
  if (N % 16 != 0 || K % 16 != 0) {
    std::cerr << "Now only support N and K is multiples of 16" << std::endl;
  }
  const int BLOCK = 128;
  int grid_x = (K + 64 - 1) / 64 + 1;
  int grid_y = (N + 128 - 1) / 128;
  dpct::dim3 grid(grid_x, grid_y);

  stream->parallel_for(
      sycl::nd_range<3>(grid * sycl::range<3>(1, 1, BLOCK),
                        sycl::range<3>(1, 1, BLOCK)),
      [=](sycl::nd_item<3> item_ct1) {
        rearrange_kn_weight_as_n32k16_order_ldg16_kernel<FType>(
            B, B_scale, B_zero, B_result, B_scale_result, B_zero_result, K, N,
            N_32align, item_ct1);
      });
}
}  // namespace allspark

void rearrange_kn_weight_as_n32k16_order(
    torch::Tensor const& b_qweight, torch::Tensor const& b_scales,
    std::optional<torch::Tensor> const& b_zeros, bool has_zp,
    torch::Tensor& b_qweight_reorder, torch::Tensor& b_scales_reorder,
    std::optional<torch::Tensor> const& b_zeros_reorder, const int64_t K,
    const int64_t N, const int64_t N_32align) {
  // Verify device and strides
  TORCH_CHECK(b_qweight.device().is_cuda(), "b_qweight is not on GPU");
  TORCH_CHECK(b_qweight.is_contiguous(), "b_qweight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(b_qweight_reorder.device().is_cuda(),
              "b_qweight_reorder is not on GPU");
  TORCH_CHECK(b_qweight_reorder.is_contiguous(),
              "b_qweight_reorder is not contiguous");

  TORCH_CHECK(b_scales_reorder.device().is_cuda(),
              "b_scales_reorder is not on GPU");
  TORCH_CHECK(b_scales_reorder.is_contiguous(),
              "b_scales_reorder is not contiguous");

  if (has_zp) {
    TORCH_CHECK(b_zeros.value().device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.value().is_contiguous(), "b_zeros is not contiguous");

    TORCH_CHECK(b_zeros_reorder.value().device().is_cuda(),
                "b_zeros_reorder is not on GPU");
    TORCH_CHECK(b_zeros_reorder.value().is_contiguous(),
                "b_zeros_reorder is not contiguous");
  }

  const uint8_t* matB = reinterpret_cast<const uint8_t*>(b_qweight.data_ptr());
  const void* b_scale = b_scales.data_ptr();
  const void* b_zero = has_zp ? b_zeros.value().data_ptr() : nullptr;

  uint8_t* matB_reorder =
      reinterpret_cast<uint8_t*>(b_qweight_reorder.data_ptr());
  void* b_scale_reorder = b_scales_reorder.data_ptr();
  void* b_zero_reorder = has_zp ? b_zeros_reorder.value().data_ptr() : nullptr;

  dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  if (b_scales.dtype() == at::ScalarType::Half) {
    allspark::rearrange_kn_weight_as_n32k16_order_ldg16<sycl::half>(
        matB, reinterpret_cast<const sycl::half*>(b_scale),
        reinterpret_cast<const sycl::half*>(b_zero), matB_reorder,
        reinterpret_cast<sycl::half*>(b_scale_reorder),
        reinterpret_cast<sycl::half*>(b_zero_reorder), K, N, N_32align, stream);
  } else if (b_scales.dtype() == at::ScalarType::BFloat16) {
    allspark::rearrange_kn_weight_as_n32k16_order_ldg16<
        sycl::ext::oneapi::bfloat16>(
        matB, reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(b_scale),
        reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(b_zero),
        matB_reorder,
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(b_scale_reorder),
        reinterpret_cast<sycl::ext::oneapi::bfloat16*>(b_zero_reorder), K, N,
        N_32align, stream);
  }
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("rearrange_kn_weight_as_n32k16_order",
         &rearrange_kn_weight_as_n32k16_order);
}
