#include "fmha_xe2.h"
#include "chunk_prefill.h"

namespace vllm::xpu::attn {

// template void fmha_kernel_impl<true>(
//     sycl::queue& queue,
//     const at::Tensor& query,
//     const at::Tensor& key_cache,
//     const at::Tensor& value_cache,
//     at::Tensor& out,
//     const at::Tensor& block_table,
//     const at::Tensor& cu_seqlens_q,
//     const at::Tensor& cu_seqlens_k,
//     int max_seqlen_q,
//     int max_seqlen_k,
//     float k_scale,
//     float v_scale,
//     float sm_scale,
//     std::optional<const at::Tensor>& sm_sink_,
//     int window_size_left,
//     int window_size_right,
//     bool is_varlen,
//     bool is_causal,
//     bool is_local,
//     bool is_sink,
//     bool is_fp8kv);

void chunk_prefill_xe2_impl(
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
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv) {
  fmha_kernel_impl_true(
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
      is_causal,
      is_local,
      is_sink,
      is_fp8kv);
}

}  // namespace vllm::xpu::attn