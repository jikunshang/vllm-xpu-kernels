#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <iostream>
#include <cmath>

constexpr uint64_t THREADS_PER_EXPERT = 512;
// threshold must match the dispatch logic in run_cutlass_moe_mm_sm90()
constexpr int SWAP_AB_THRESHOLD = 64;

template <bool SWAP_AB>
void compute_problem_sizes(const int32_t* __restrict__ topk_ids,
                                      int32_t* problem_sizes1,
                                      int32_t* problem_sizes2,
                                      int32_t* atomic_buffer,
                                      const int topk_length, const int n,
                                      const int k,
                                      const sycl::nd_item<3> &item_ct1) {
  int expert_id = item_ct1.get_group(2);

  int occurrences = 0;
  for (int i = item_ct1.get_local_id(2); i < topk_length;
       i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
      &atomic_buffer[expert_id], occurrences);
  /*
  DPCT1065:371: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    if constexpr (!SWAP_AB) {
      problem_sizes1[expert_id * 3] = final_occurrences;
      problem_sizes1[expert_id * 3 + 1] = 2 * n;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = final_occurrences;
      problem_sizes2[expert_id * 3 + 1] = k;
      problem_sizes2[expert_id * 3 + 2] = n;
    } else {
      problem_sizes1[expert_id * 3] = 2 * n;
      problem_sizes1[expert_id * 3 + 1] = final_occurrences;
      problem_sizes1[expert_id * 3 + 2] = k;
      problem_sizes2[expert_id * 3] = k;
      problem_sizes2[expert_id * 3 + 1] = final_occurrences;
      problem_sizes2[expert_id * 3 + 2] = n;
    }
  }
}

void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts, const int topk_length) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += topk_length > SWAP_AB_THRESHOLD ? problem_sizes1[i * 3]
                                                  : problem_sizes1[i * 3 + 1];
    expert_offsets[i + 1] = tot_offset;
  }
}

void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* blockscale_offsets, int32_t* atomic_buffer, const int num_experts,
    const int topk_length) {
  int32_t tot_offset = 0;
  int32_t tot_offset_round = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    int32_t cur_offset = topk_length > SWAP_AB_THRESHOLD
                             ? problem_sizes1[i * 3]
                             : problem_sizes1[i * 3 + 1];
    atomic_buffer[i] = tot_offset;
    tot_offset += cur_offset;
    expert_offsets[i + 1] = tot_offset;
    tot_offset_round += (cur_offset + (128 - 1)) / 128 * 128;
    blockscale_offsets[i + 1] = tot_offset_round;
  }
}

void compute_arg_sorts(const int32_t* __restrict__ topk_ids,
                                  const int32_t* __restrict__ expert_offsets,
                                  int32_t* input_permutation,
                                  int32_t* output_permutation,
                                  int32_t* atomic_buffer, const int topk_length,
                                  const int topk,
                                  const sycl::nd_item<3> &item_ct1) {
  int const blk_expert_id = item_ct1.get_group(2);
  int const num_experts = item_ct1.get_group_range(2);
  int32_t const num_tokens = expert_offsets[num_experts];

  for (int i = item_ct1.get_local_id(2); i < topk_length;
       i += THREADS_PER_EXPERT) {
    int const expert_id = topk_ids[i];
    if (expert_id == -1 && item_ct1.get_group(2) == 0) {
      // output_permutation is used to re-order the moe outputs. It is
      // used as c2 = c2[c_map], where c2 is a torch.tensor that is the
      // output of the cutlass kernels and c_map is the output_permutation.
      // c2 is initialized to zeros, therefore by setting the output_permutation
      // to num_tokens, we are guaranteed to fill the moe outputs to zero
      // for "invalid" topk_ids.
      output_permutation[i] = num_tokens;
    } else if (expert_id == blk_expert_id) {
      int start =
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              &atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  int num_threads = min(THREADS_PER_EXPERT, topk_ids.numel());

  if (topk_ids.numel() > SWAP_AB_THRESHOLD) {
    /*
    DPCT1049:228: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto topk_ids_data_ptr_ct0 =
          static_cast<const int32_t*>(topk_ids.data_ptr());
      auto problem_sizes1_data_ptr_ct1 =
          static_cast<int32_t*>(problem_sizes1.data_ptr());
      auto problem_sizes2_data_ptr_ct2 =
          static_cast<int32_t*>(problem_sizes2.data_ptr());
      auto atomic_buffer_data_ptr_ct3 =
          static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct4 = topk_ids.numel();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_experts) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         compute_problem_sizes<false>(
                             topk_ids_data_ptr_ct0, problem_sizes1_data_ptr_ct1,
                             problem_sizes2_data_ptr_ct2,
                             atomic_buffer_data_ptr_ct3, topk_ids_numel_ct4, n,
                             k, item_ct1);
                       });
    });
  } else {
    /*
    DPCT1049:229: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto topk_ids_data_ptr_ct0 =
          static_cast<const int32_t*>(topk_ids.data_ptr());
      auto problem_sizes1_data_ptr_ct1 =
          static_cast<int32_t*>(problem_sizes1.data_ptr());
      auto problem_sizes2_data_ptr_ct2 =
          static_cast<int32_t*>(problem_sizes2.data_ptr());
      auto atomic_buffer_data_ptr_ct3 =
          static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct4 = topk_ids.numel();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_experts) *
                                             sycl::range<3>(1, 1, num_threads),
                                         sycl::range<3>(1, 1, num_threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         compute_problem_sizes<true>(
                             topk_ids_data_ptr_ct0, problem_sizes1_data_ptr_ct1,
                             problem_sizes2_data_ptr_ct2,
                             atomic_buffer_data_ptr_ct3, topk_ids_numel_ct4, n,
                             k, item_ct1);
                       });
    });
  }

  if (blockscale_offsets.has_value()) {
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto problem_sizes1_data_ptr_ct0 =
          static_cast<const int32_t*>(problem_sizes1.data_ptr());
      auto expert_offsets_data_ptr_ct1 =
          static_cast<int32_t*>(expert_offsets.data_ptr());
      auto blockscale_offsets_value_data_ptr_ct2 =
          static_cast<int32_t*>(blockscale_offsets.value().data_ptr());
      auto atomic_buffer_data_ptr_ct3 =
          static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct5 = topk_ids.numel();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            compute_expert_blockscale_offsets(
                problem_sizes1_data_ptr_ct0, expert_offsets_data_ptr_ct1,
                blockscale_offsets_value_data_ptr_ct2,
                atomic_buffer_data_ptr_ct3, num_experts, topk_ids_numel_ct5);
          });
    });
  } else {
    ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
      auto problem_sizes1_data_ptr_ct0 =
          static_cast<const int32_t*>(problem_sizes1.data_ptr());
      auto expert_offsets_data_ptr_ct1 =
          static_cast<int32_t*>(expert_offsets.data_ptr());
      auto atomic_buffer_data_ptr_ct2 =
          static_cast<int32_t*>(atomic_buffer.data_ptr());
      auto topk_ids_numel_ct4 = topk_ids.numel();

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            compute_expert_offsets(
                problem_sizes1_data_ptr_ct0, expert_offsets_data_ptr_ct1,
                atomic_buffer_data_ptr_ct2, num_experts, topk_ids_numel_ct4);
          });
    });
  }
  /*
  DPCT1049:227: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
    auto topk_ids_data_ptr_ct0 =
        static_cast<const int32_t*>(topk_ids.data_ptr());
    auto expert_offsets_data_ptr_ct1 =
        static_cast<const int32_t*>(expert_offsets.data_ptr());
    auto input_permutation_data_ptr_ct2 =
        static_cast<int32_t*>(input_permutation.data_ptr());
    auto output_permutation_data_ptr_ct3 =
        static_cast<int32_t*>(output_permutation.data_ptr());
    auto atomic_buffer_data_ptr_ct4 =
        static_cast<int32_t*>(atomic_buffer.data_ptr());
    auto topk_ids_numel_ct5 = topk_ids.numel();
    auto topk_ids_size_ct6 = topk_ids.size(1);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_experts) *
                                           sycl::range<3>(1, 1, num_threads),
                                       sycl::range<3>(1, 1, num_threads)),
                     [=](sycl::nd_item<3> item_ct1) {
                       compute_arg_sorts(
                           topk_ids_data_ptr_ct0, expert_offsets_data_ptr_ct1,
                           input_permutation_data_ptr_ct2,
                           output_permutation_data_ptr_ct3,
                           atomic_buffer_data_ptr_ct4, topk_ids_numel_ct5,
                           topk_ids_size_ct6, item_ct1);
                     });
  });
}

void compute_pplx_data(int32_t* expert_offsets,
                                  int32_t* problem_sizes1,
                                  int32_t* problem_sizes2,
                                  const int32_t* __restrict__ expert_num_tokens,
                                  const int padded_m, const int n,
                                  const int k, const sycl::nd_item<3> &item_ct1) {
  int expert_idx = item_ct1.get_local_id(2);

  expert_offsets[expert_idx] = expert_idx * padded_m;
  problem_sizes1[expert_idx * 3] = expert_num_tokens[expert_idx];
  problem_sizes1[expert_idx * 3 + 1] = 2 * n;
  problem_sizes1[expert_idx * 3 + 2] = k;
  problem_sizes2[expert_idx * 3] = expert_num_tokens[expert_idx];
  problem_sizes2[expert_idx * 3 + 1] = k;
  problem_sizes2[expert_idx * 3 + 2] = n;
}

void get_cutlass_pplx_moe_mm_data_caller(torch::Tensor& expert_offsets,
                                         torch::Tensor& problem_sizes1,
                                         torch::Tensor& problem_sizes2,
                                         const torch::Tensor& expert_num_tokens,
                                         const int64_t num_local_experts,
                                         const int64_t padded_m,
                                         const int64_t n, const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(expert_offsets.device().index());

  /*
  DPCT1049:230: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  ((sycl::queue*)(stream))->submit([&](sycl::handler& cgh) {
    auto expert_offsets_data_ptr_ct0 =
        static_cast<int32_t*>(expert_offsets.data_ptr());
    auto problem_sizes1_data_ptr_ct1 =
        static_cast<int32_t*>(problem_sizes1.data_ptr());
    auto problem_sizes2_data_ptr_ct2 =
        static_cast<int32_t*>(problem_sizes2.data_ptr());
    auto expert_num_tokens_data_ptr_ct3 =
        static_cast<const int32_t*>(expert_num_tokens.data_ptr());

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_local_experts),
                                       sycl::range<3>(1, 1, num_local_experts)),
                     [=](sycl::nd_item<3> item_ct1) {
                       compute_pplx_data(expert_offsets_data_ptr_ct0,
                                         problem_sizes1_data_ptr_ct1,
                                         problem_sizes2_data_ptr_ct2,
                                         expert_num_tokens_data_ptr_ct3,
                                         padded_m, n, k, item_ct1);
                     });
  });
}