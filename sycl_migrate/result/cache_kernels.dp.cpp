#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.dp.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  dpct::memcpy_direction memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = dpct::device_to_device;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = dpct::device_to_host;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = dpct::host_to_device;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    /*
    DPCT1124:22: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    stream->memcpy(dst_ptr + dst_offset, src_ptr + src_offset,
                   block_size_in_bytes);
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block,
                                   const sycl::nd_item<3> &item_ct1) {
  const int layer_idx = item_ct1.get_group(2);
  const int pair_idx = item_ct1.get_group(1);

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = item_ct1.get_local_id(2); i < numel_per_block;
       i += item_ct1.get_local_range(2)) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = item_ct1.get_local_id(2); i < numel_per_block;
       i += item_ct1.get_local_range(2)) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block, const sycl::nd_item<3> &item_ct1) {
  const int layer_idx = item_ct1.get_group(2);
  const int pair_idx = item_ct1.get_group(1);
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = item_ct1.get_local_id(2); i < mem_footprint_per_block;
       i += item_ct1.get_local_range(2)) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping) {
  int num_layers = key_caches.size();
  TORCH_CHECK(num_layers == value_caches.size());
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }

  // block_mapping is a 2D tensor with shape (num_pairs, 2).
  int num_pairs = block_mapping.size(0);

  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(cache_device);

  // Launch the kernel.
  const int numel_per_block = key_caches[0][0].numel();
  dpct::dim3 grid(num_layers, num_pairs);
  dpct::dim3 block(std::min(1024, numel_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
      key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
        vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
            key_cache_ptrs_tensor.data_ptr<int64_t>(),
            value_cache_ptrs_tensor.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), numel_per_block);
      }));
}

// copy blocks kernel for MLA (assumes a joint KV-cache)
void copy_blocks_mla(std::vector<torch::Tensor> const& kv_caches,
                     const torch::Tensor& block_mapping) {
  int num_layers = kv_caches.size();
  if (num_layers == 0) {
    return;
  }
  torch::Device cache_device = kv_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda(), "kv_cache must be on CUDA");

  std::vector<int64_t> cache_ptrs(num_layers);
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(kv_caches[layer_idx].data_ptr());
  }
  torch::Tensor cache_ptrs_tensor =
      torch::from_blob(cache_ptrs.data(), {num_layers}, torch::kInt64)
          .to(cache_device);

  int num_pairs = block_mapping.size(0);
  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  int mem_footprint_per_block = kv_caches[0].stride(0);
  dpct::dim3 grid(num_layers, num_pairs);
  dpct::dim3 block(std::min(1024, mem_footprint_per_block));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
      kv_caches[0].scalar_type(), "copy_blocks_mla_kernel", ([&] {
        vllm::copy_blocks_mla_kernel<scalar_t><<<grid, block, 0, stream>>>(
            cache_ptrs_tensor.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), mem_footprint_per_block);
      }));
}

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale,
    const sycl::nd_item<3> &item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range(2)) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const sycl::nd_item<3> &item_ct1) {
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range(2)) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_key_value_idx = block_idx * block_stride +
                                      block_offset * page_stride +
                                      head_idx * head_stride + head_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_value_idx] = tgt_key;
      value_cache[tgt_key_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         ,
    const sycl::nd_item<3> &item_ct1//
) {
  const int64_t token_idx = item_ct1.get_group(2);
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset,
                  const sycl::nd_item<3> &item_ct1) {
    for (int i = item_ct1.get_local_id(2); i < size;
         i += item_ct1.get_local_range(2)) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
/*
DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                        \
 stream->submit([&](sycl::handler& cgh) {                                      \
  auto key_data_ptr_ct0 = reinterpret_cast<KV_T*>(key.data_ptr());             \
  auto value_data_ptr_ct1 = reinterpret_cast<KV_T*>(value.data_ptr());         \
  auto key_cache_data_ptr_ct2 =                                                \
      reinterpret_cast<CACHE_T*>(key_cache.data_ptr());                        \
  auto value_cache_data_ptr_ct3 =                                              \
      reinterpret_cast<CACHE_T*>(value_cache.data_ptr());                      \
  auto slot_mapping_data_ptr_int64_t_ct4 = slot_mapping.data_ptr<int64_t>();   \
  auto key_stride_ct5 = key_stride;                                            \
  auto value_stride_ct6 = value_stride;                                        \
  auto num_heads_ct7 = num_heads;                                              \
  auto head_size_ct8 = head_size;                                              \
  auto block_size_ct9 = block_size;                                            \
  auto x_ct10 = x;                                                             \
  auto k_scale_data_ptr_ct11 =                                                 \
      reinterpret_cast<const float*>(k_scale.data_ptr());                      \
  auto v_scale_data_ptr_ct12 =                                                 \
      reinterpret_cast<const float*>(v_scale.data_ptr());                      \
                                                                               \
  cgh.parallel_for(                                                            \
      sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) { \
       vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>(                \
           key_data_ptr_ct0, value_data_ptr_ct1, key_cache_data_ptr_ct2,       \
           value_cache_data_ptr_ct3, slot_mapping_data_ptr_int64_t_ct4,        \
           key_stride_ct5, value_stride_ct6, num_heads_ct7, head_size_ct8,     \
           block_size_ct9, x_ct10, k_scale_data_ptr_ct11,                      \
           v_scale_data_ptr_ct12, item_ct1);                                   \
      });                                                                      \
 });

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dpct::dim3 grid(num_tokens);
  dpct::dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE)
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
/*
DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)                  \
 stream->submit([&](sycl::handler& cgh) {                                      \
  auto key_data_ptr_ct0 = reinterpret_cast<KV_T*>(key.data_ptr());             \
  auto value_data_ptr_ct1 = reinterpret_cast<KV_T*>(value.data_ptr());         \
  auto key_cache_data_ptr_ct2 =                                                \
      reinterpret_cast<CACHE_T*>(key_cache.data_ptr());                        \
  auto value_cache_data_ptr_ct3 =                                              \
      reinterpret_cast<CACHE_T*>(value_cache.data_ptr());                      \
  auto slot_mapping_data_ptr_int64_t_ct4 = slot_mapping.data_ptr<int64_t>();   \
  auto block_stride_ct5 = block_stride;                                        \
  auto page_stride_ct6 = page_stride;                                          \
  auto head_stride_ct7 = head_stride;                                          \
  auto key_stride_ct8 = key_stride;                                            \
  auto value_stride_ct9 = value_stride;                                        \
  auto num_heads_ct10 = num_heads;                                             \
  auto head_size_ct11 = head_size;                                             \
  auto block_size_ct12 = block_size;                                           \
  auto k_scale_data_ptr_ct13 =                                                 \
      reinterpret_cast<const float*>(k_scale.data_ptr());                      \
  auto v_scale_data_ptr_ct14 =                                                 \
      reinterpret_cast<const float*>(v_scale.data_ptr());                      \
                                                                               \
  cgh.parallel_for(                                                            \
      sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) { \
       vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>(          \
           key_data_ptr_ct0, value_data_ptr_ct1, key_cache_data_ptr_ct2,       \
           value_cache_data_ptr_ct3, slot_mapping_data_ptr_int64_t_ct4,        \
           block_stride_ct5, page_stride_ct6, head_stride_ct7, key_stride_ct8, \
           value_stride_ct9, num_heads_ct10, head_size_ct11, block_size_ct12,  \
           k_scale_data_ptr_ct13, v_scale_data_ptr_ct14, item_ct1);            \
      });                                                                      \
 });

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dpct::dim3 grid(num_tokens);
  dpct::dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
/*
DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)                     \
 stream->submit([&](sycl::handler& cgh) {                                      \
  auto kv_c_data_ptr_ct0 = reinterpret_cast<KV_T*>(kv_c.data_ptr());           \
  auto k_pe_data_ptr_ct1 = reinterpret_cast<KV_T*>(k_pe.data_ptr());           \
  auto kv_cache_data_ptr_ct2 =                                                 \
      reinterpret_cast<CACHE_T*>(kv_cache.data_ptr());                         \
  auto slot_mapping_data_ptr_int64_t_ct3 = slot_mapping.data_ptr<int64_t>();   \
  auto block_stride_ct4 = block_stride;                                        \
  auto entry_stride_ct5 = entry_stride;                                        \
  auto kv_c_stride_ct6 = kv_c_stride;                                          \
  auto k_pe_stride_ct7 = k_pe_stride;                                          \
  auto kv_lora_rank_ct8 = kv_lora_rank;                                        \
  auto pe_dim_ct9 = pe_dim;                                                    \
  auto block_size_ct10 = block_size;                                           \
  auto scale_data_ptr_ct11 = reinterpret_cast<const float*>(scale.data_ptr()); \
                                                                               \
  cgh.parallel_for(                                                            \
      sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) { \
       vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>(             \
           kv_c_data_ptr_ct0, k_pe_data_ptr_ct1, kv_cache_data_ptr_ct2,        \
           slot_mapping_data_ptr_int64_t_ct3, block_stride_ct4,                \
           entry_stride_ct5, kv_c_stride_ct6, k_pe_stride_ct7,                 \
           kv_lora_rank_ct8, pe_dim_ct9, block_size_ct10, scale_data_ptr_ct11, \
           item_ct1);                                                          \
      });                                                                      \
 });

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  dpct::dim3 grid(num_tokens);
  dpct::dim3 block(std::min(kv_lora_rank, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                             CALL_CONCAT_AND_CACHE_MLA);
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride,
                                   const sycl::nd_item<3> &item_ct1) {
  const int64_t block_idx = item_ct1.get_group(2);
  for (int i = item_ct1.get_local_id(2); i < block_stride;
       i += item_ct1.get_local_range(2)) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype) {
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dpct::dim3 grid(num_blocks);
  dpct::dim3 block(std::min(block_stride, int64_t(512)));
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, sycl::ext::oneapi::bfloat16,
                       vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(sycl::ext::oneapi::bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, sycl::ext::oneapi::bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(sycl::ext::oneapi::bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t>
void gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRIES...]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts, const sycl::nd_item<3> &item_ct1) {  // Optional: starting offsets per
                                               // batch

  const int64_t bid = item_ct1.get_group(2);  // Batch ID
  const int32_t num_splits = item_ct1.get_group_range(1);
  const int32_t split = item_ct1.get_group(1);
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_blocks = cuda_utils::ceil_div(seq_len, block_size);
  const int32_t split_blocks = cuda_utils::ceil_div(tot_blocks, num_splits);

  const int32_t split_start = split * split_blocks;
  const int32_t split_end = dpct::min((split + 1) * split_blocks, tot_blocks);

  const bool is_active_split = (split_start < tot_blocks);
  const bool is_last_split = (split_end == tot_blocks);

  if (!is_active_split) return;

  int32_t full_blocks_end = split_end;
  int32_t partial_block_size = 0;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on (seq_starts[bid] /
  // page_size)
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = 0;
  if (seq_starts != nullptr) {
    offset = seq_starts[bid] / block_size;
  }
  const int32_t* batch_block_table = block_table + batch_offset + offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  if (is_last_split) {
    partial_block_size = seq_len % block_size;
    if (partial_block_size) full_blocks_end -= 1;
  }

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst,
                        const sycl::nd_item<3> &item_ct1) {
    for (int i = item_ct1.get_local_id(2); i < entry_size;
         i += item_ct1.get_local_range(2))
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < full_blocks_end; ++pid) {
    auto block_id = batch_block_table[pid];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * block_size * dst_entry_stride;
    for (int eid = 0; eid < block_size; ++eid) {
      copy_entry(block_start_ptr + eid * cache_entry_stride,
                 block_dst_ptr + eid * dst_entry_stride);
    }
  }

  if (partial_block_size) {
    auto block_id = batch_block_table[full_blocks_end];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + full_blocks_end * block_size * dst_entry_stride;
    for (int eid = 0; eid < partial_block_size; ++eid) {
      copy_entry(block_start_ptr + eid * cache_entry_stride,
                 block_dst_ptr + eid * dst_entry_stride);
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
/*
DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
#define CALL_GATHER_CACHE(CPY_DTYPE)                                           \
 stream->submit([&](sycl::handler& cgh) {                                      \
  auto src_cache_data_ptr_ct0 =                                                \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr());                      \
  auto dst_data_ptr_ct1 = reinterpret_cast<CPY_DTYPE*>(dst.data_ptr());        \
  auto block_table_data_ptr_int32_t_ct2 = block_table.data_ptr<int32_t>();     \
  auto cu_seq_lens_data_ptr_int32_t_ct3 = cu_seq_lens.data_ptr<int32_t>();     \
  auto block_size_ct4 = block_size;                                            \
  auto entry_size_ct5 = entry_size;                                            \
  auto block_table_stride_ct6 = block_table_stride;                            \
  auto cache_block_stride_ct7 = cache_block_stride;                            \
  auto cache_entry_stride_ct8 = cache_entry_stride;                            \
  auto dst_entry_stride_ct9 = dst_entry_stride;                                \
  auto seq_starts_ptr_ct10 = seq_starts_ptr;                                   \
                                                                               \
  cgh.parallel_for(                                                            \
      sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) { \
       vllm::gather_cache<CPY_DTYPE>(                                          \
           src_cache_data_ptr_ct0, dst_data_ptr_ct1,                           \
           block_table_data_ptr_int32_t_ct2, cu_seq_lens_data_ptr_int32_t_ct3, \
           block_size_ct4, entry_size_ct5, block_table_stride_ct6,             \
           cache_block_stride_ct7, cache_entry_stride_ct8,                     \
           dst_entry_stride_ct9, seq_starts_ptr_ct10, item_ct1);               \
      });                                                                      \
 });

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dpct::dim3 grid(batch_size, num_splits);
  dpct::dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}
