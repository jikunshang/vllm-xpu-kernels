/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "attention_kernels.dp.hpp"
#include "cuda_compat.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

/*
DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                   \
   VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                       \
       ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,          \
                                               BLOCK_SIZE, NUM_THREADS,        \
                                               KV_DTYPE, IS_BLOCK_SPARSE>),    \
       shared_mem_size);                                                       \
   stream->submit([&](sycl::handler& cgh) {                                    \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(                     \
          sycl::range<1>(shared_mem_size), cgh);                               \
      sycl::local_accessor<Q_vec[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD], 0>   \
          q_vecs_acc_ct1(cgh);                                                 \
      sycl::local_accessor<float, 1> red_smem_acc_ct1(                         \
          sycl::range<1>(2 * NUM_WARPS), cgh);                                 \
                                                                               \
      auto out_ptr_ct0 = out_ptr;                                              \
      auto query_ptr_ct1 = query_ptr;                                          \
      auto key_cache_ptr_ct2 = key_cache_ptr;                                  \
      auto value_cache_ptr_ct3 = value_cache_ptr;                              \
      auto num_kv_heads_ct4 = num_kv_heads;                                    \
      auto scale_ct5 = scale;                                                  \
      auto block_tables_ptr_ct6 = block_tables_ptr;                            \
      auto seq_lens_ptr_ct7 = seq_lens_ptr;                                    \
      auto max_num_blocks_per_seq_ct8 = max_num_blocks_per_seq;                \
      auto alibi_slopes_ptr_ct9 = alibi_slopes_ptr;                            \
      auto q_stride_ct10 = q_stride;                                           \
      auto kv_block_stride_ct11 = kv_block_stride;                             \
      auto kv_head_stride_ct12 = kv_head_stride;                               \
      auto k_scale_ptr_ct13 = k_scale_ptr;                                     \
      auto v_scale_ptr_ct14 = v_scale_ptr;                                     \
      auto tp_rank_ct15 = tp_rank;                                             \
      auto blocksparse_local_blocks_ct16 = blocksparse_local_blocks;           \
      auto blocksparse_vert_stride_ct17 = blocksparse_vert_stride;             \
      auto blocksparse_block_size_ct18 = blocksparse_block_size;               \
      auto blocksparse_head_sliding_step_ct19 = blocksparse_head_sliding_step; \
                                                                               \
      cgh.parallel_for(                                                        \
          sycl::nd_range<3>(grid * block, block),                              \
          [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {   \
             vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,            \
                                             BLOCK_SIZE, NUM_THREADS,          \
                                             KV_DTYPE, IS_BLOCK_SPARSE>(       \
                 out_ptr_ct0, query_ptr_ct1, key_cache_ptr_ct2,                \
                 value_cache_ptr_ct3, num_kv_heads_ct4, scale_ct5,             \
                 block_tables_ptr_ct6, seq_lens_ptr_ct7,                       \
                 max_num_blocks_per_seq_ct8, alibi_slopes_ptr_ct9,             \
                 q_stride_ct10, kv_block_stride_ct11, kv_head_stride_ct12,     \
                 k_scale_ptr_ct13, v_scale_ptr_ct14, tp_rank_ct15,             \
                 blocksparse_local_blocks_ct16, blocksparse_vert_stride_ct17,  \
                 blocksparse_block_size_ct18,                                  \
                 blocksparse_head_sliding_step_ct19, item_ct1,                 \
                 dpct_local_acc_ct1                                            \
                     .get_multi_ptr<sycl::access::decorated::no>()             \
                     .get(),                                                   \
                 q_vecs_acc_ct1,                                               \
                 red_smem_acc_ct1.get_multi_ptr<sycl::access::decorated::no>() \
                     .get());                                                  \
          });                                                                  \
   });

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  /*
  DPCT1083:11: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dpct::dim3 grid(num_heads, num_seqs, 1);
  dpct::dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 32:
      LAUNCH_PAGED_ATTENTION_V1(32);
      break;
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,        \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  if (is_block_sparse) {                                                   \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);       \
  } else {                                                                 \
    CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);      \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);

  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
