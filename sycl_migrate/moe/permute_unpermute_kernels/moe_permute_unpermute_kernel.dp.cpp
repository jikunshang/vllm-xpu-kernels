#define DPCT_COMPAT_RT_VERSION 12080

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "moe_permute_unpermute_kernel.h"

// moe_permute kernels require at least CUDA 12.0
#if defined(DPCT_COMPAT_RT_VERSION) && (DPCT_COMPAT_RT_VERSION >= 12000)

// CubKeyValueSorter definition begin
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

int CubKeyValueSorter::expertsToBits(int num_experts) {
  // Max value we represent is V = num_experts + (num_experts - 1) = 2 *
  // num_experts - 1 The maximum number of bits is therefore floor(log2(V)) + 1
  return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts), num_bits_(expertsToBits(num_experts)) {}

void CubKeyValueSorter::updateNumExperts(int const num_experts) {
  num_experts_ = num_experts;
  num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs,
                                           int const num_experts) {
  int num_bits = expertsToBits(num_experts);
  size_t required_storage = 0;
  int* null_int = nullptr;
  cub::DeviceRadixSort::SortPairs(nullptr, required_storage, null_int, null_int,
                                  null_int, null_int, num_key_value_pairs, 0,
                                  num_bits);

  //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64,
  //   4, 3, 0 The required_storage seems to vary between 0 and 1 for the same
  //   inputs
  if (required_storage == 0) {
    required_storage = 1;
  }
  return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size,
                            int const* keys_in, int* keys_out,
                            int const* values_in, int* values_out,
                            size_t const num_key_value_pairs,
                            dpct::queue_ptr stream) {
  size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
  size_t actual_ws_size = workspace_size;

  TORCH_CHECK(expected_ws_size <= workspace_size,
              "[CubKeyValueSorter::run] The allocated workspace is too small "
              "to run this problem.");
  cub::DeviceRadixSort::SortPairs(workspace, actual_ws_size, keys_in, keys_out,
                                  values_in, values_out, num_key_value_pairs, 0,
                                  num_bits_, stream);
}
// CubKeyValueSorter definition end

static inline size_t pad_to_multiple_of_16(size_t const& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}
template <class T>
inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length,
                                                      T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

// Calculates the start offset of the tokens for a given expert. The last
// element is the total number of valid tokens
void computeExpertFirstTokenOffsetKernel(
    int const* sorted_experts, int64_t const sorted_experts_len,
    int const num_experts, int64_t* expert_first_token_offset,
    const sycl::nd_item<3> &item_ct1) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);

  // Note that expert goes [0, num_experts] (inclusive) because we want a count
  // for the total number of active tokens at the end of the scan.
  if (expert >= num_experts + 1) {
    return;
  }
  expert_first_token_offset[expert] =
      findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   dpct::queue_ptr stream) {
  int const num_entries = num_experts + 1;
  int const threads = std::min(1024, num_entries);
  int const blocks = (num_entries + threads - 1) / threads;

  /*
  DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, threads),
                                         sycl::range<3>(1, 1, threads)),
                       [=](sycl::nd_item<3> item_ct1) {
                         computeExpertFirstTokenOffsetKernel(
                             sorted_indices, total_indices, num_experts,
                             expert_first_token_offset, item_ct1);
                       });
}

void sortAndScanExpert(int* expert_for_source_row, const int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
                       dpct::queue_ptr stream) {
  int64_t const expanded_num_rows = static_cast<int64_t>(k) * num_rows;
  // We need to use the full num_experts because that is the sentinel value used
  // by topk for disabled experts
  sorter.updateNumExperts(num_experts);
  size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(
      sorter.getWorkspaceSize(expanded_num_rows, num_experts));
  sorter.run((void*)sorter_ws, sorter_ws_size_bytes, expert_for_source_row,
             permuted_experts, source_rows, permuted_rows, expanded_num_rows,
             stream);
  computeExpertFirstTokenOffset(permuted_experts, expanded_num_rows,
                                num_experts_per_node, expert_first_token_offset,
                                stream);
}

void preprocessTopkIdKernel(int* topk_id_ptr, int size,
                                       const int* expert_map_ptr,
                                       int num_experts,
                                       const sycl::nd_item<3> &item_ct1,
                                       uint8_t *dpct_local) {
  auto tidx = item_ct1.get_local_id(2);
  auto bidx = item_ct1.get_group(2);
  auto offset = bidx * item_ct1.get_local_range(2);
  auto bound =
      dpct::min((unsigned int)(offset + item_ct1.get_local_range(2)), size);
  auto smem_expert_map = (int*)dpct_local;
  // store expert_map in smem
  for (int i = tidx; i < num_experts; i += item_ct1.get_local_range(2)) {
    smem_expert_map[i] = expert_map_ptr[i];
  }
  /*
  DPCT1065:295: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  // query global expert id in expert map.
  // if global expert id = -1 in exert map, plus n_expert
  // else set global expert id = exert map[global expert id]
  if (offset + tidx < bound) {
    auto topk_id = topk_id_ptr[offset + tidx];
    auto local_expert_idx = smem_expert_map[topk_id];
    if (local_expert_idx == -1) {
      topk_id += num_experts;
    } else {
      topk_id = local_expert_idx;
    }
    sycl::group_barrier(item_ct1.get_sub_group());
    topk_id_ptr[offset + tidx] = topk_id;
  }
}
void preprocessTopkIdLauncher(int* topk_id_ptr, int size,
                              const int* expert_map_ptr, int num_experts,
                              dpct::queue_ptr stream) {
  int block = std::min(size, 1024);
  int grid = (size + block - 1) / block;
  /*
  DPCT1083:24: The size of local memory in the migrated code may be different
  from the original code. Check that the allocated memory size in the migrated
  code is correct.
  */
  int smem_size = (num_experts) * sizeof(int);
  /*
  DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  stream->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(smem_size), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, grid) * sycl::range<3>(1, 1, block),
            sycl::range<3>(1, 1, block)),
        [=](sycl::nd_item<3> item_ct1) {
          preprocessTopkIdKernel(
              topk_id_ptr, size, expert_map_ptr, num_experts, item_ct1,
              dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                  .get());
        });
  });
}

template <bool ALIGN_BLOCK_SIZE>
void getMIndicesKernel(int64_t* expert_first_token_offset,
                                  int64_t* align_expert_first_token_offset,
                                  int* m_indices, const int num_local_expert,
                                  const int align_block_size,
                                  const sycl::nd_item<3> &item_ct1,
                                  uint8_t *dpct_local) {
  int eidx = item_ct1.get_group(2);
  int tidx = item_ct1.get_local_id(2);
  auto smem_expert_first_token_offset = (int64_t*)dpct_local;
  for (int i = tidx; i <= num_local_expert; i += item_ct1.get_local_range(2)) {
    /*
    DPCT1098:296: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    smem_expert_first_token_offset[tidx] = *(expert_first_token_offset + i);
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  auto last_token_offset = smem_expert_first_token_offset[eidx + 1];
  auto first_token_offset = smem_expert_first_token_offset[eidx];
  int n_token_in_expert = last_token_offset - first_token_offset;

  if constexpr (ALIGN_BLOCK_SIZE) {
    n_token_in_expert = (n_token_in_expert + align_block_size - 1) /
                        align_block_size * align_block_size;
    // round up to ALIGN_BLOCK_SIZE
    int64_t accumulate_align_offset = 0;
    for (int i = 1; i <= eidx + 1; i++) {
      int n_token = smem_expert_first_token_offset[i] -
                    smem_expert_first_token_offset[i - 1];
      accumulate_align_offset =
          accumulate_align_offset + (n_token + align_block_size - 1) /
                                        align_block_size * align_block_size;
      if (i == eidx) {
        first_token_offset = accumulate_align_offset;
      }
      // last block store align_expert_first_token_offset
      if (eidx == num_local_expert - 1 && item_ct1.get_local_id(2) == 0) {
        align_expert_first_token_offset[i] = accumulate_align_offset;
      }
    }
  }
  for (int idx = tidx; idx < n_token_in_expert;
       idx += item_ct1.get_local_range(2)) {
    // update m_indice with expert id
    m_indices[first_token_offset + idx] = eidx;
  }
}

void getMIndices(int64_t* expert_first_token_offset,
                 int64_t* align_expert_first_token_offset, int* m_indices,
                 int num_local_expert, const int align_block_size,
                 dpct::queue_ptr stream) {
  int block = 256;
  int grid = num_local_expert;
  int smem_size = sizeof(int64_t) * (num_local_expert + 1);
  if (align_block_size == -1) {
    getMIndicesKernel<false><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  } else {
    getMIndicesKernel<true><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  }
}

#endif
