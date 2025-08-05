#pragma once

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

#ifdef USE_ROCM
  #include "amd/quant_utils.cuh"
#endif

// Determines the preferred FP8 type for the current platform.
// Note that for CUDA this just returns true,
// but on ROCm it will check device props.
static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

namespace vllm {

__dpct_inline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? sycl::bit_cast<float>(dpct::atomic_fetch_max<
                                    sycl::access::address_space::generic_space>(
                  (int*)addr, sycl::bit_cast<int>(value)))
            : sycl::bit_cast<float>(dpct::atomic_fetch_min<
                                    sycl::access::address_space::generic_space>(
                  (unsigned int*)addr, sycl::bit_cast<unsigned int>(value)));

  return old;
}

template <bool is_scale_inverted, typename fp8_type>
__dpct_inline__ fp8_type scaled_fp8_conversion(float const val,
                                               float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
#ifndef USE_ROCM
  return static_cast<fp8_type>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t, typename fp8_type>
void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *cache) {
  int64_t i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = fmaxf(tmp, sycl::fabs(x));
    i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
  }
  cache[item_ct1.get_local_id(2)] = tmp;

  /*
  DPCT1065:311: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // Now perform parallel reduction within the thread block
  int ib = item_ct1.get_local_range(2) / 2;
  while (ib != 0) {
    if (item_ct1.get_local_id(2) < ib && cache[item_ct1.get_local_id(2) + ib] >
                                             cache[item_ct1.get_local_id(2)]) {
      cache[item_ct1.get_local_id(2)] = cache[item_ct1.get_local_id(2) + ib];
    }
    /*
    DPCT1118:31: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:312: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (item_ct1.get_local_id(2) == 0) {
    atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t>
float thread_max_vec(scalar_t const* __restrict__ input,
                                int64_t const num_elems, int const tid,
                                int const step) {
  constexpr size_t VEC_SIZE = 16;
  using scalarxN_t = vec_n_t<scalar_t, VEC_SIZE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<scalarxN_t const*>(input);

  // num_elems / VEC_SIZE (which is 16)
  int64_t const num_vec_elems = num_elems >> 4;
  float absmax_val = 0.0f;

#pragma unroll
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    scalarxN_t in_vec = vectorized_in[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      absmax_val = fmaxf(absmax_val, fabsf(in_vec.val[j]));
    }
  }

  // Handle the remaining elements if num_elems is not divisible by VEC_SIZE
  for (int64_t i = num_vec_elems * VEC_SIZE + tid; i < num_elems; i += step) {
    absmax_val = fmaxf(absmax_val, fabsf(input[i]));
  }

  return absmax_val;
}

template <typename scalar_t, bool is_scale_inverted, typename fp8_type>
void scaled_fp8_conversion_vec(fp8_type* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          float const scale,
                                          int64_t const num_elems,
                                          int const tid, int const step) {
  constexpr size_t VEC_SIZE = 16;
  using scalarxN_t = vec_n_t<scalar_t, VEC_SIZE>;
  using float8xN_t = q8_n_t<fp8_type, VEC_SIZE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<scalarxN_t const*>(input);
  auto* vectorized_out = reinterpret_cast<float8xN_t*>(out);

  // num_elems / VEC_SIZE (which is 16)
  int64_t const num_vec_elems = num_elems >> 4;

#pragma unroll
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    scalarxN_t in_vec = vectorized_in[i];
    float8xN_t out_vec;

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out_vec.val[j] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
          static_cast<float>(in_vec.val[j]), scale);
    }
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by VEC_SIZE
  for (int64_t i = num_vec_elems * VEC_SIZE + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(input[i]), scale);
  }
}

}  // namespace vllm
