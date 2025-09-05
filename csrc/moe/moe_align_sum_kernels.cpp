#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>

#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include "dispatch_utils.h"
#include "utils.h"

#define WARP_SIZE 32
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {
template <typename scalar_t>
class moe_align_block_size_kernel {
 public:
  moe_align_block_size_kernel(const scalar_t* __restrict__ topk_ids_,
                              int32_t* __restrict__ sorted_token_ids_,
                              int32_t* __restrict__ expert_ids_,
                              int32_t* __restrict__ total_tokens_post_pad_,
                              int32_t num_experts_, int32_t padded_num_experts_,
                              int32_t experts_per_warp_, int32_t block_size_,
                              size_t numel_, int32_t* __restrict__ cumsum_,
                              int32_t max_num_tokens_padded_,
                              sycl::local_accessor<int32_t, 1> shared_count_)
      : topk_ids(topk_ids_),
        sorted_token_ids(sorted_token_ids),
        expert_ids(expert_ids_),
        total_tokens_post_pad(total_tokens_post_pad_),
        num_experts(num_experts_),
        padded_num_experts(padded_num_experts_),
        experts_per_warp(experts_per_warp_),
        block_size(block_size_),
        numel(numel_),
        cumsum(cumsum_),
        max_num_tokens_padded(max_num_tokens_padded_),
        shared_count(shared_count_) {}

  void operator() [[intel::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    auto shared_counts = (int32_t*)shared_count.get_pointer();

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
    DPCT1065:301: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:302: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int expert_count = 0;
    int expert_id = item_ct1.get_local_id(2);
    if (expert_id < num_experts) {
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    cumsum_val = sycl::exclusive_scan_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(), expert_count, 0,
        sycl::plus<>());
    if (expert_id <= num_experts) {
      cumsum[expert_id] = cumsum_val;
    }

    if (expert_id == num_experts) {
      *total_tokens_post_pad = cumsum_val;
    }

    /*
    DPCT1065:303: Consider replacing sycl::nd_item::barrier() with
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

 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ expert_ids;
  int32_t* __restrict__ total_tokens_post_pad;
  int32_t num_experts;
  int32_t padded_num_experts;
  int32_t experts_per_warp;
  int32_t block_size;
  size_t numel;
  int32_t* __restrict__ cumsum;
  int32_t max_num_tokens_padded;
  sycl::local_accessor<int32_t, 1> shared_count;
};

template <typename scalar_t>
class count_and_sort_expert_tokens_kernel {
 public:
  count_and_sort_expert_tokens_kernel(const scalar_t* __restrict__ topk_ids_,
                                      int32_t* __restrict__ sorted_token_ids_,
                                      int32_t* __restrict__ cumsum_buffer_,
                                      size_t numel_)
      : topk_ids(topk_ids_),
        sorted_token_ids(sorted_token_ids_),
        cumsum_buffer(cumsum_buffer_),
        numel(numel_) {}

  void operator()(const sycl::nd_item<3>& item_ct1) const {
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

 private:
  const scalar_t* __restrict__ topk_ids;
  int32_t* __restrict__ sorted_token_ids;
  int32_t* __restrict__ cumsum_buffer;
  size_t numel;
};

template <typename scalar_t, int TOPK>
class moe_sum_kernel {
 private:
  scalar_t* output;       // [..., d]
  const scalar_t* input;  // [..., topk, d]
  int d;

 public:
  moe_sum_kernel(scalar_t* output, const scalar_t* input, int d)
      : output(output), input(input), d(d) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_idx = item.get_group(0);
    for (int64_t idx = item.get_local_id(0); idx < d;
         idx += item.get_local_range(0)) {
      scalar_t x = 0.0;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        x += input[token_idx * TOPK * d + k * d + idx];
      }
      output[token_idx * d + idx] = x;
    }
  }
};

}  // namespace moe
}  // namespace vllm

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad) {
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();

  int64_t padded_num_experts =
      ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  sycl::range<3> grid(1, 1, 1);
  sycl::range<3> block(1, 1, threads);

  // BlockScan uses 1024 threads and assigns one thread per expert.
  TORCH_CHECK(padded_num_experts < 1024,
              "padded_num_experts must be less than 1024");

  VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(
      topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        using sycl_t = vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        // calc needed amount of shared mem for `cumsum` tensors
        auto options_int =
            torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
        torch::Tensor cumsum_buffer =
            torch::empty({num_experts + 1}, options_int);
        bool small_batch_expert_mode = false;
        // (topk_ids.numel() < 1024) && (num_experts <= 64);

        if (small_batch_expert_mode) {
        } else {
          size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
          queue.submit([&](sycl::handler& cgh) {
            size_t shared_mem_size =
                num_warps * experts_per_warp * sizeof(int32_t);
            sycl::local_accessor<int> shared_counts(
                sycl::range<1>(shared_mem_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block),
                vllm::moe::moe_align_block_size_kernel<sycl_t>(
                    (sycl_t*)topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                    padded_num_experts, experts_per_warp, block_size,
                    topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(),
                    sorted_token_ids.size(0), shared_counts));
          });

          const int block_threads = std::min(256, (int)threads);
          const int num_blocks =
              (topk_ids.numel() + block_threads - 1) / block_threads;
          const int max_blocks = 65535;
          const int actual_blocks = std::min(num_blocks, max_blocks);
          sycl::range<3> grid_1(1, 1, actual_blocks);
          sycl::range<3> block_1(1, 1, block_threads);

          queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid_1 * block_1, block_1),
                vllm::moe::count_and_sort_expert_tokens_kernel<sycl_t>(
                    (sycl_t*)topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(),
                    cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel()));
          });
        }
      });
}

void moe_sum(torch::Tensor& input,   // [num_tokens, topk, hidden_size]
             torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(hidden_size, 1024));
  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);
  auto& queue = vllm::xpu::vllmGetQueue();

  switch (topk) {
    case 2:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 2>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    case 3:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 3>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    case 4:
      VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                           vllm::moe::moe_sum_kernel<scalar_t, 4>(
                               output.data_ptr<scalar_t>(),
                               input.data_ptr<scalar_t>(), hidden_size));
        });
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}
