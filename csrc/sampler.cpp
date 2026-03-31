// SPDX-License-Identifier: Apache-2.0
#include <sycl/sycl.hpp>
#include <algorithm>
#include "utils.h"
#include "dispatch_utils.h"

namespace vllm {

template <typename scalar_t>
class apply_repetition_penalties_kernel {
 public:
  apply_repetition_penalties_kernel(
      scalar_t* __restrict__ logits,
      const bool* __restrict__ prompt_mask,
      const bool* __restrict__ output_mask,
      const scalar_t* __restrict__ repetition_penalties,
      const int num_seqs,
      const int vocab_size,
      const int tile_size)
      : logits_(logits),
        prompt_mask_(prompt_mask),
        output_mask_(output_mask),
        repetition_penalties_(repetition_penalties),
        num_seqs_(num_seqs),
        vocab_size_(vocab_size),
        tile_size_(tile_size) {}

  void operator() [[sycl::reqd_sub_group_size(32)]] (
      const sycl::nd_item<3>& item_ct1) const {
    // Each block handles one sequence and a tile of vocab
    const int seq_idx = item_ct1.get_group(2);
    if (seq_idx >= num_seqs_) return;

    const int tile_idx = item_ct1.get_group(1);
    const int tile_start = tile_idx * tile_size_;
    const int tile_end =
        sycl::min((const int)(tile_start + tile_size_), vocab_size_);

    // Load repetition penalty for this sequence
    const scalar_t penalty = repetition_penalties_[seq_idx];

    // Each thread processes multiple vocab items within the tile
    for (int vocab_idx = tile_start + item_ct1.get_local_id(2);
         vocab_idx < tile_end;
         vocab_idx += item_ct1.get_local_range(2)) {
      const int64_t idx =
          static_cast<int64_t>(seq_idx) * vocab_size_ + vocab_idx;
      const bool is_repeated = prompt_mask_[idx] || output_mask_[idx];
      if (is_repeated) {
        scalar_t logit = logits_[idx];
        if (logit > scalar_t(0)) {
          logits_[idx] = logit / penalty;
        } else {
          logits_[idx] = logit * penalty;
        }
      }
    }
  }

 private:
  scalar_t* __restrict__ logits_;
  const bool* __restrict__ prompt_mask_;
  const bool* __restrict__ output_mask_;
  const scalar_t* __restrict__ repetition_penalties_;
  const int num_seqs_;
  const int vocab_size_;
  const int tile_size_;
};

}  // namespace vllm

void apply_repetition_penalties_(
    torch::Tensor& logits,
    const torch::Tensor& prompt_mask,
    const torch::Tensor& output_mask,
    const torch::Tensor& repetition_penalties) {
  CHECK_DEVICE(logits);
  CHECK_DEVICE(prompt_mask);
  CHECK_DEVICE(output_mask);
  CHECK_DEVICE(repetition_penalties);

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(prompt_mask.is_contiguous(), "prompt_mask must be contiguous");
  TORCH_CHECK(output_mask.is_contiguous(), "output_mask must be contiguous");
  TORCH_CHECK(
      repetition_penalties.is_contiguous(),
      "repetition_penalties must be contiguous");

  int vocab_size = logits.size(-1);
  int num_seqs = logits.size(0);

  if (num_seqs == 0) return;

  // Get device properties for determining tile configuration
  // Use a heuristic similar to CUDA: aim for good occupancy
  constexpr int max_threads_per_block = 1024;
  int tile_num = std::max(
      1, (vocab_size + max_threads_per_block - 1) / max_threads_per_block);
  int tile_size = (vocab_size + tile_num - 1) / tile_num;

  sycl::range<3> grid(1, tile_num, num_seqs);
  sycl::range<3> block(1, 1, std::min(tile_size, max_threads_per_block));

  at::DeviceGuard device_guard(logits.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalties_kernel", [&] {
        using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<3>(grid * block, block),
              vllm::apply_repetition_penalties_kernel<sycl_t>(
                  reinterpret_cast<sycl_t*>(logits.data_ptr<scalar_t>()),
                  prompt_mask.data_ptr<bool>(),
                  output_mask.data_ptr<bool>(),
                  reinterpret_cast<const sycl_t*>(
                      repetition_penalties.data_ptr<scalar_t>()),
                  num_seqs,
                  vocab_size,
                  tile_size));
        });
      });
}
