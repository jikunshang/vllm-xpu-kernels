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

template void policy_dispatch<chunk_policy_head64, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

template void policy_dispatch<chunk_policy_head64, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

}  // namespace vllm::xpu::attn