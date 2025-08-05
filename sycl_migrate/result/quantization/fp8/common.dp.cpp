#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.dp.hpp"
#include <dpct/dpl_utils.hpp>

#include "dispatch_utils.h"

#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

template <typename scalar_t, typename fp8_type>
void scaled_fp8_quant_kernel(fp8_type* __restrict__ out,
                                        const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale,
                                        int64_t num_elems,
                                        const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(
      out, input, inverted_scale, num_elems, tid,
      item_ct1.get_local_range(2) * item_ct1.get_group_range(2));
}

template <typename scalar_t, typename fp8_type>
void dynamic_per_token_scaled_fp8_quant_kernel(
    fp8_type* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    const int hidden_size, const sycl::nd_item<3> &item_ct1,  &reduceStorage,
    float &token_scale) {
  int const tid = item_ct1.get_local_id(2);
  int const token_idx = item_ct1.get_group(2);

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  fp8_type* __restrict__ token_output = &out[offset];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 32-byte and 16-byte addresses respectively.
  bool const can_vectorize = hidden_size % 16 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid,
                                item_ct1.get_local_range(2));
  } else {
    for (int i = tid; i < hidden_size; i += item_ct1.get_local_range(2)) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = sycl::fmax(absmax_val, sycl::fabs((float)x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 256>;

  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage)
          .Reduce(absmax_val, sycl::maximum<>{}, item_ct1.get_local_range(2));

  if (tid == 0) {
    if (scale_ub) {
      token_scale =
          sycl::fmin((float)block_absmax_val_maybe, (float)(*scale_ub));
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scale[token_idx] = token_scale;
  }
  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(token_output, token_input,
                                               token_scale, hidden_size, tid,
                                               item_ct1.get_local_range(2));
  } else {
    for (int i = tid; i < hidden_size; i += item_ct1.get_local_range(2)) {
      token_output[i] = scaled_fp8_conversion<false, fp8_type>(
          static_cast<float>(token_input[i]), token_scale);
    }
  }
}

}  // namespace vllm

void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dpct::dim3 const grid(num_tokens);
  dpct::dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                              torch::Tensor const& input,  // [..., d]
                              torch::Tensor& scale)        // [1]
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int const block_size = 256;
  int const num_tokens = input.numel() / input.size(-1);
  int const num_elems = input.numel();
  dpct::dim3 const grid(num_tokens);
  dpct::dim3 const block(block_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::segmented_max_reduction<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(scale.data_ptr<float>(),
                                               input.data_ptr<scalar_t>(),
                                               num_elems);
              vllm::scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), num_elems);
            });
      });
}

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out,          // [..., d]
    torch::Tensor const& input,  // [..., d]
    torch::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  int const block_size = 256;
  dpct::dim3 const grid(num_tokens);
  dpct::dim3 const block(std::min(hidden_size, block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type", [&] {
              vllm::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), scales.data_ptr<float>(),
                      input.data_ptr<scalar_t>(),
                      scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                           : nullptr,
                      hidden_size);
            });
      });
}
