#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include <cmath>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {

template <typename scalar_t>
void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t* __restrict__ cumsum, int32_t max_num_tokens_padded,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local,  &temp_storage) {
  auto shared_counts = (int32_t*)dpct_local;

  // Initialize sorted_token_ids with numel
  for (size_t it = item_ct1.get_local_id(2); it < max_num_tokens_padded;
       it += item_ct1.get_local_range(2)) {
    sorted_token_ids[it] = numel;
  }

  const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  /*
  DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  const size_t tid = item_ct1.get_local_id(2);
  const size_t stride = item_ct1.get_local_range(2);

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int32_t, 1024>;

  int expert_count = 0;
  int expert_id = item_ct1.get_local_id(2);
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    *total_tokens_post_pad = cumsum_val;
  }

  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < num_experts) {
    for (int i = cumsum[item_ct1.get_local_id(2)];
         i < cumsum[item_ct1.get_local_id(2) + 1]; i += block_size) {
      expert_ids[i / block_size] = item_ct1.get_local_id(2);
    }
  }

  // Fill remaining expert_ids with 0
  const size_t fill_start_idx =
      cumsum[num_experts] / block_size + item_ct1.get_local_id(2);
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size;
       i += item_ct1.get_local_range(2)) {
    expert_ids[i] = 0;
  }
}

template <typename scalar_t>
void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer,
    size_t numel, const sycl::nd_item<3> &item_ct1) {
  const size_t tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
  const size_t stride =
      item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t, int TOPK>
void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d,
    const sycl::nd_item<3> &item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  for (int64_t idx = item_ct1.get_local_id(2); idx < d;
       idx += item_ct1.get_local_range(2)) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename scalar_t>
void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, size_t numel, int32_t max_num_tokens_padded,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  // Initialize sorted_token_ids with numel
  for (size_t it = item_ct1.get_local_id(2); it < max_num_tokens_padded;
       it += item_ct1.get_local_range(2)) {
    sorted_token_ids[it] = numel;
  }

  const size_t tid = item_ct1.get_local_id(2);
  const size_t stride = item_ct1.get_local_range(2);

  auto shared_mem = (int32_t*)dpct_local;
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(item_ct1.get_local_id(2) + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(item_ct1.get_local_id(2) + 1) * num_experts + topk_ids[i]];
  }

  /*
  DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < num_experts) {
    tokens_cnts[item_ct1.get_local_id(2)] = 0;
    for (int i = 1; i <= item_ct1.get_local_range(2); ++i) {
      tokens_cnts[i * num_experts + item_ct1.get_local_id(2)] +=
          tokens_cnts[(i - 1) * num_experts + item_ct1.get_local_id(2)];
    }
  }

  /*
  DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(
              tokens_cnts[item_ct1.get_local_range(2) * num_experts + i - 1],
              block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  /*
  DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < num_experts) {
    for (int i = cumsum[item_ct1.get_local_id(2)];
         i < cumsum[item_ct1.get_local_id(2) + 1]; i += block_size) {
      expert_ids[i / block_size] = item_ct1.get_local_id(2);
    }
  }

  // Fill remaining expert_ids with 0
  const size_t fill_start_idx =
      cumsum[num_experts] / block_size + item_ct1.get_local_id(2);
  const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
  for (size_t i = fill_start_idx; i < expert_ids_size;
       i += item_ct1.get_local_range(2)) {
    expert_ids[i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        tokens_cnts[item_ct1.get_local_id(2) * num_experts + expert_id] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[item_ct1.get_local_id(2) * num_experts + expert_id];
  }
}

}  // namespace moe
}  // namespace vllm

// taken from
// https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::empty({num_experts + 1}, options_int);
        bool small_batch_expert_mode =
            (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
          const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
          const int32_t shared_mem_size =
              ((threads + 1) * num_experts + (num_experts + 1)) *
              sizeof(int32_t);

          auto small_batch_expert_kernel =
              vllm::moe::moe_align_block_size_small_batch_expert_kernel<
                  scalar_t>;
          small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size,
              topk_ids.numel(), sorted_token_ids.size(0));
        } else {
          auto align_kernel = vllm::moe::moe_align_block_size_kernel<scalar_t>;

          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          size_t shared_mem_size =
              num_warps * experts_per_warp * sizeof(int32_t);

          align_kernel<<<1, threads, shared_mem_size, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              experts_ids.data_ptr<int32_t>(),
              num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
              padded_num_experts, experts_per_warp, block_size,
              topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
              sorted_token_ids.size(0));

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);

          auto sort_kernel =
              vllm::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
          sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
              topk_ids.data_ptr<scalar_t>(),
              sorted_token_ids.data_ptr<int32_t>(),
              cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
        }
      });
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dpct::dim3 grid(num_tokens);
  dpct::dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        vllm::moe::moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
