#include "csrc/utils.h"
#include "attn_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "csrc/xpu/attn/xe_2/fmha_xe2.h"
#endif

namespace vllm::xpu::attn {

void fmha_interface(
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
    bool is_paged,
    bool is_causal,
    bool is_local,
    bool is_sink,
    bool is_fp8kv) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel
    if (is_paged) {
      chunk_prefill_xe2_impl(
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
    } else {
      varlen_xe2_impl(
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
          is_sink);
    }
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
  }
}

}  // namespace vllm::xpu::attn