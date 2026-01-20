#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "chunk_prefill.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "kernel/chunk_prefill_kernel.hpp"

#include "fmha_utils.hpp"

using namespace cute;
namespace vllm::xpu::attn {

template <bool Paged>
void fmha_kernel_impl(
    sycl::queue& queue,
    const at::Tensor& query,      // [seq_q, heads, head_size]
    const at::Tensor& key_cache,  // [num_block, block_size, heads, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float k_scale,
    float v_scale,
    float sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv) {
  // general params
  int batch_size, num_heads_q, num_heads_kv, head_size;
  // additional params
  int total_seqlen_q, total_seqlen_k;
  int num_blocks, block_size, max_blocks_per_seq;
  if (is_varlen) {
    // query: [total_seq, num_heads, head_size]
    batch_size = cu_seqlens_q.numel() - 1;
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(2);
    total_seqlen_q = query.size(0);
    total_seqlen_k = key_cache.size(0);
  } else {
    // query: [batch, num_heads, seq, head_size]
    batch_size = query.size(0);
    num_heads_q = query.size(1);
    num_heads_kv = key_cache.size(1);
    head_size = query.size(3);
    max_seqlen_q = query.size(2);
    max_seqlen_k = key_cache.size(2);
  }
  if constexpr (Paged) {
    num_blocks = key_cache.size(0);
    block_size = key_cache.size(1);
    num_heads_kv = key_cache.size(2);
    max_blocks_per_seq = block_table.size(1);
    total_seqlen_k = num_blocks * block_size;
  }

  if (is_local) {
    window_size_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
    window_size_right =
        window_size_right == -1 ? max_seqlen_k : window_size_right;
    if (is_causal) {
      window_size_right = 0;
      is_causal = false;
    }
  }

  chunk_prefill_args_t<Paged> args = {
      query.data_ptr(),
      key_cache.data_ptr(),
      value_cache.data_ptr(),
      out.data_ptr(),
      Paged ? block_table.data_ptr() : nullptr,
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      k_scale,
      v_scale,
      sm_scale,
      is_sink ? sm_sink_.value().data_ptr() : nullptr,
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      max_blocks_per_seq,
      block_size,
      window_size_left,
      window_size_right,
      is_varlen,  // varlen
      is_causal,
      is_local,
      is_sink,
      is_fp8kv};

  // CutlassType cuType = aten_to_Cutlass_dtype(query);

  static constexpr int max_head_size = 256;
  TORCH_CHECK(
      head_size <= max_head_size,
      "FMHA forward only supports head dimension at most " +
          std::to_string(max_head_size));

  if (args.head_size <= HEAD_SIZE_LIMIT_0) {
    policy_dispatch<chunk_policy_head64>(
        queue, query.scalar_type(), key_cache.scalar_type(), args);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_1) {
    policy_dispatch<chunk_policy_head96>(
        queue, query.scalar_type(), key_cache.scalar_type(), args);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_2) {
    policy_dispatch<chunk_policy_head128>(
        queue, query.scalar_type(), key_cache.scalar_type(), args);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_3) {
    policy_dispatch<chunk_policy_head192>(
        queue, query.scalar_type(), key_cache.scalar_type(), args);
  } else if (args.head_size <= HEAD_SIZE_LIMIT_4) {
    policy_dispatch<chunk_policy_head256>(
        queue, query.scalar_type(), key_cache.scalar_type(), args);
  } else {
    TORCH_CHECK(false, "Unsupported head size for fmha");
  }
}

template void fmha_kernel_impl<true>(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float k_scale,
    float v_scale,
    float sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv);

template void fmha_kernel_impl<false>(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float k_scale,
    float v_scale,
    float sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_varlen,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv);

void fmha_kernel_impl_true(
    sycl::queue& queue,
    const at::Tensor& query,      // [batch, heads, seq, head_size]
    const at::Tensor& key_cache,  // [batch, heads, seq, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float k_scale,
    float v_scale,
    float sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv) {
  fmha_kernel_impl<true>(
      queue,
      query,
      key_cache,
      value_cache,
      out,
      block_table,
      cu_seqlens_q,
      cu_seqlens_k,
      max_seqlen_q,
      max_seqlen_k,
      k_scale,
      v_scale,
      sm_scale,
      sm_sink_,
      window_size_left,
      window_size_right,
      true,
      is_causal,
      is_local,
      is_sink,
      is_fp8kv);
}

void fmha_kernel_impl_false(
    sycl::queue& queue,
    const at::Tensor& query,      // [batch, heads, seq, head_size]
    const at::Tensor& key_cache,  // [batch, heads, seq, head_size]
    const at::Tensor& value_cache,
    at::Tensor& out,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float k_scale,
    float v_scale,
    float sm_scale,
    std::optional<const at::Tensor>& sm_sink_,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv) {
  fmha_kernel_impl<false>(
      queue,
      query,
      key_cache,
      value_cache,
      out,
      block_table,
      cu_seqlens_q,
      cu_seqlens_k,
      max_seqlen_q,
      max_seqlen_k,
      k_scale,
      v_scale,
      sm_scale,
      sm_sink_,
      window_size_left,
      window_size_right,
      true,
      is_causal,
      is_local,
      is_sink,
      is_fp8kv);
};
}  // namespace vllm::xpu::attn