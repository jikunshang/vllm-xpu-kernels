#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "quant_conversions.dp.hpp"
#include <dpct/dpl_utils.hpp>

#ifndef USE_ROCM
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            const sycl::nd_item<3> &item_ct1,  &reduceStore,
                            float &s_rms,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);
  // sum of squares
  float ss = 0.0f;

  for (auto i = item_ct1.get_local_id(2); i < hidden_size;
       i += item_ct1.get_local_range(2)) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;

  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  if (item_ct1.get_local_id(2) == 0) {
    s_rms = sycl::rsqrt(ss / hidden_size + epsilon);
  }
  /*
  DPCT1065:346: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size, const sycl::nd_item<3> &item_ct1,  &reduceStore,
    float &s_token_scale,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);
  ;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  float block_absmax_val_maybe = 0.0f;
  for (auto i = item_ct1.get_local_id(2); i < hidden_size;
       i += item_ct1.get_local_range(2)) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    block_absmax_val_maybe = sycl::fmax(block_absmax_val_maybe, sycl::fabs(x));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;

  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, sycl::maximum<>{},
                  item_ct1.get_local_range(2));

  if (item_ct1.get_local_id(2) == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = sycl::min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[item_ct1.get_group(2)] = scale;  // Global output store
  }
  /*
  DPCT1065:347: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  *token_scale = s_token_scale;
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               const sycl::nd_item<3> &item_ct1,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = item_ct1.get_local_id(2); i < hidden_size;
       i += item_ct1.get_local_range(2)) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

namespace vectorized {

// Compute 1.0/rms(input)
// hidden_size must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            const sycl::nd_item<3> &item_ct1,  &reduceStore,
                            float &s_rms,
                            scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (auto i = item_ct1.get_local_id(2); i < num_vec_elems;
       i += item_ct1.get_local_range(2)) {
    vec4_t<scalar_t> in = vec_input[i];

    vec4_t<float> x;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      x.val[j] = static_cast<float>(in.val[j]);
    }

    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] += static_cast<float>(r.val[j]);
      }
    }

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      ss += x.val[j] * x.val[j];
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;

  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  if (item_ct1.get_local_id(2) == 0) {
    s_rms = sycl::rsqrt(ss / hidden_size + epsilon);
  }
  /*
  DPCT1065:348: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  *rms = s_rms;
}

// Vectorized version of vllm::compute_dynamic_per_token_scales
// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    int32_t const hidden_size, const sycl::nd_item<3> &item_ct1,  &reduceStore,
    float &s_token_scale,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems = hidden_size >> 2;
  float block_absmax_val_maybe = 0.0f;

#pragma unroll 4
  for (auto i = item_ct1.get_local_id(2); i < num_vec_elems;
       i += item_ct1.get_local_range(2)) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      x.val[j] = static_cast<float>(in.val[j]);
    }

    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] += static_cast<float>(r.val[j]);
      }
    }

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      block_absmax_val_maybe =
          fmaxf(block_absmax_val_maybe,
                fabs(static_cast<scalar_t>(x.val[j] * rms) * w.val[j]));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;

  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, sycl::maximum<>{},
                  item_ct1.get_local_range(2));

  if (item_ct1.get_local_id(2) == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = sycl::min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor<scalar_out_t>::val());
    s_token_scale = scale;                 // shared memory store
    all_token_scales[item_ct1.get_group(2)] = scale;  // global output store
  }
  /*
  DPCT1065:349: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  *token_scale = s_token_scale;
}

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false>
void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               const sycl::nd_item<3> &item_ct1,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset =
      item_ct1.get_group(2) * static_cast<int64_t>(hidden_size);
  ;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  const int VEC_SIZE = 4;
  int32_t const num_vec_elems = hidden_size >> 2;

// TODO(luka/varun) extract into type-agnostic vectorized quant function to
//  replace scaled_fp8_conversion_vec
#pragma unroll 4
  for (auto i = item_ct1.get_local_id(2); i < num_vec_elems;
       i += item_ct1.get_local_range(2)) {
    vec4_t<scalar_t> const in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      x.val[j] = static_cast<float>(in.val[j]);
    }

    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        x.val[j] += static_cast<float>(r.val[j]);
      }
// Update residual
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        r.val[j] = static_cast<scalar_t>(x.val[j]);
      }
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out.val[j] = ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(
          static_cast<scalar_t>(x.val[j] * rms) * w.val[j], scale);
    }
    vec_output[i] = out;
  }
}

}  // namespace vectorized

}  // namespace vllm
