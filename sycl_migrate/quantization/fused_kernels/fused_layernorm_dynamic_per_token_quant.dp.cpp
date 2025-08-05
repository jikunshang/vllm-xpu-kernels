
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../dispatch_utils.h"
#include "layernorm_utils.dp.hpp"
#include "quant_conversions.dp.hpp"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    const sycl::nd_item<3> &item_ct1,  &reduceStore, float &s_rms,
    float &s_token_scale,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, item_ct1, reduceStore, s_rms,
      residual);

  // Compute scale
  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size, item_ct1,
      reduceStore, s_token_scale, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, item_ct1,
        residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, token_scale, hidden_size, item_ct1, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
void rms_norm_dynamic_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon, int32_t const hidden_size,
    const sycl::nd_item<3> &item_ct1,  &reduceStore, float &s_rms,
    float &s_token_scale,
    scalar_t* __restrict__ residual = nullptr) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, hidden_size,
        item_ct1, reduceStore, s_rms, s_token_scale, residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, item_ct1, reduceStore,
                                            s_rms, residual);
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub, hidden_size, item_ct1,
      reduceStore, s_token_scale, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, item_ct1,
        residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms, token_scale, hidden_size, item_ct1, residual);
  }
}
}  // namespace vllm

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> const& scale_ub,
    std::optional<at::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  dpct::dim3 grid(num_tokens);
  dpct::dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  if (residual.has_value()) {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        true>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, residual->data_ptr<scalar_in_t>());
        });

  } else {
    VLLM_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        false>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, hidden_size, nullptr);
        });
  }
}

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> scale_ub, std::optional<at::Tensor> residual) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }
  TORCH_CHECK(scales.dtype() == torch::kFloat32);

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_token_quant_dispatch", [&] {
        rms_norm_dynamic_per_token_quant_dispatch<scalar_t>(
            out, input, weight, scales, var_epsilon, scale_ub, residual);
      });
}
