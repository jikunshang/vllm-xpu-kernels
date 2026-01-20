#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>
#include <torch/all.h>

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
    bool is_fp8kv);

template <bool Paged>
struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  float k_scale = 1.0;
  float v_scale = 1.0;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  bool is_fp8kv = false;

  static constexpr bool is_page = Paged;
};

template <class FMHAKernel, bool isVarLen>
struct KernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::FMHAProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  template <typename Args>
  ProblemShapeType initialize(const Args& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = args.batch_size;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q =
        args.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv =
        args.num_heads_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk =
        args.head_size;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo =
        args.head_size;

    if constexpr (isVarLen) {
      batch = shape_init.batch = 1;
      shape_init.seq_len_qo = args.total_seqlen_q;
      shape_init.seq_len_kv = args.total_seqlen_k;

      shape.seq_len_qo =
          cutlass::fmha::collective::VariableLength{args.max_queries};
      shape.seq_len_qo.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_q);
      shape.seq_len_kv =
          cutlass::fmha::collective::VariableLength{args.max_keys};
      shape.seq_len_kv.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_k);
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = args.max_queries;
      shape.seq_len_kv = shape_init.seq_len_kv = args.max_keys;
    }

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    stride_Q = cutlass::make_cute_packed_stride(
        StrideQ{},
        cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    stride_K = cutlass::make_cute_packed_stride(
        StrideK{},
        cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
    stride_V = cutlass::make_cute_packed_stride(
        StrideV{},
        cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));

    return shape;
  }

  template <typename Args>
  cutlass::Status
  run(sycl::queue& queue,
      const Args& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<ElementQ*>(args.query),
         stride_Q,
         reinterpret_cast<ElementK*>(args.key),
         stride_K,
         reinterpret_cast<ElementV*>(args.value),
         stride_V,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementQ*>(args.sm_sink)},
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, workspace.get());

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    run(queue, params);

    return cutlass::Status::kSuccess;
  }

  static void run(sycl::queue& queue, typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event =
        compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
            policy, queue, params);

    EventManager::getInstance().addEvent(event);
  }
};

template <
    typename ElementQ,
    typename ElementK,
    typename ElementV,
    typename ElementO>
struct FMHAElementTypes {
  using Q = ElementQ;
  using K = ElementK;
  using V = ElementV;
  using O = ElementO;
};

template <
    typename StrideQ,
    typename StrideK,
    typename StrideV,
    typename StrideO>
struct FMHAStrides {
  using Q = StrideQ;
  using K = StrideK;
  using V = StrideV;
  using O = StrideO;
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    typename ElementTypes,         // FMHAElementTypes<>
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct FMHAConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, typename ElementTypes::Q>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <typename Args, class Scheduler, bool Causal, bool Local, bool Sink>
  static void run(sycl::queue& queue, const Args& args) {
    constexpr bool VarLen = true;
    constexpr bool Paged = Args::is_page;
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<VarLen>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ =
        decltype(make_dummy_tensor(typename ElementTypes::Q{}, StrideQ{}));
    using TensorK =
        decltype(make_dummy_tensor(typename ElementTypes::K{}, StrideK{}));
    using TensorV =
        decltype(make_dummy_tensor(typename ElementTypes::V{}, StrideV{}));
    using TensorO =
        decltype(make_dummy_tensor(typename ElementTypes::O{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Paged,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        Sink,
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    KernelLauncher<FMHAKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  template <typename Args>
  static void kernel_dispatch_optimized(sycl::queue& queue, const Args& args) {
    // 3个布尔值 = 8种组合，使用索引直接跳转
    constexpr auto dispatch_table = []() {
      using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler;
      std::array<void (*)(sycl::queue&, const Args&), 8> table{};
      table[0] = &run<Args, Scheduler, false, false, false>;
      table[1] = &run<Args, Scheduler, false, false, true>;
      table[2] = &run<Args, Scheduler, false, true, false>;
      table[3] = &run<Args, Scheduler, false, true, true>;
      table[4] = &run<Args, Scheduler, true, false, false>;
      table[5] = &run<Args, Scheduler, true, false, true>;
      table[6] = &run<Args, Scheduler, true, true, false>;
      table[7] = &run<Args, Scheduler, true, true, true>;
      return table;
    }();

    int index = (args.is_causal ? 4 : 0) + (args.is_local ? 2 : 0) +
                (args.is_sink ? 1 : 0);

    dispatch_table[index](queue, args);
  }

  template <typename Args, bool... Bs>
  static void kernel_dispatch(sycl::queue& queue, const Args& args) {
    return run<
        Args,
        cutlass::fmha::kernel::XeFHMAIndividualTileScheduler,
        Bs...>(queue, args);
  }

  template <typename Args, bool... Bs, typename... Ts>
  static void
  kernel_dispatch(sycl::queue& queue, const Args& args, bool b, Ts... ts) {
    if (b) {
      kernel_dispatch<Args, Bs..., true>(queue, args, ts...);
    } else {
      kernel_dispatch<Args, Bs..., false>(queue, args, ts...);
    }
  }
};

template <typename QueryType, typename KeyType>
struct TypePair {
  using Query = QueryType;
  using Key = KeyType;
};

using HalfQ_HalfK = TypePair<half_t, half_t>;

template <typename TPair>
struct TypeDispatcher;

template <typename chunk_policy, typename Args>
void policy_dispatch(
    sycl::queue& queue,
    c10::ScalarType query_dtype,
    c10::ScalarType key_dtype,
    const Args& args) {
  const int PipelineStages = 2;

  if (query_dtype == torch::kHalf) {
    if (key_dtype == torch::kFloat8_e5m2) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          FMHAElementTypes<half_t, float_e5m2_t, float_e5m2_t, half_t>>::
          kernel_dispatch_optimized(queue, args);
      // kernel_dispatch(
      //     queue, args, args.is_causal, args.is_local, args.is_sink);
    } else if (key_dtype == torch::kFloat8_e4m3fn) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          FMHAElementTypes<half_t, float_e4m3_t, float_e4m3_t, half_t>>::
          kernel_dispatch_optimized(queue, args);
      // kernel_dispatch(
      //     queue, args, args.is_causal, args.is_local, args.is_sink);
    }
    return FMHAConfig<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        FMHAElementTypes<half_t, half_t, half_t, half_t>>::
        kernel_dispatch_optimized(queue, args);
    // kernel_dispatch(
    //     queue, args, args.is_causal, args.is_local, args.is_sink);
  } else if (query_dtype == torch::kBFloat16) {
    if (key_dtype == torch::kFloat8_e5m2) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          FMHAElementTypes<
              bfloat16_t,
              float_e5m2_t,
              float_e5m2_t,
              bfloat16_t>>::kernel_dispatch_optimized(queue, args);
      // kernel_dispatch(
      //     queue, args, args.is_causal, args.is_local, args.is_sink);
    } else if (key_dtype == torch::kFloat8_e4m3fn) {
      return FMHAConfig<
          typename chunk_policy::ShapeQK,
          typename chunk_policy::ShapePV,
          typename chunk_policy::ShapeOut,
          typename chunk_policy::SubgroupLayoutQK,
          void,
          PipelineStages,
          FMHAElementTypes<
              bfloat16_t,
              float_e4m3_t,
              float_e4m3_t,
              bfloat16_t>>::kernel_dispatch_optimized(queue, args);
      // kernel_dispatch(
      //     queue, args, args.is_causal, args.is_local, args.is_sink);
    }
    return FMHAConfig<
        typename chunk_policy::ShapeQK,
        typename chunk_policy::ShapePV,
        typename chunk_policy::ShapeOut,
        typename chunk_policy::SubgroupLayoutQK,
        void,
        PipelineStages,
        FMHAElementTypes<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t>>::
        kernel_dispatch_optimized(queue, args);
    // kernel_dispatch(
    //     queue, args, args.is_causal, args.is_local, args.is_sink);
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Current cutlass kernel only support half/bf16 data type.");
  }
}

extern template void
policy_dispatch<chunk_policy_head64, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

extern template void
policy_dispatch<chunk_policy_head64, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

extern template void
policy_dispatch<chunk_policy_head96, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

extern template void
policy_dispatch<chunk_policy_head96, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

extern template void
policy_dispatch<chunk_policy_head128, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

extern template void
policy_dispatch<chunk_policy_head128, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

extern template void
policy_dispatch<chunk_policy_head192, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

extern template void
policy_dispatch<chunk_policy_head192, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

extern template void
policy_dispatch<chunk_policy_head256, chunk_prefill_args_t<true>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<true>&);

extern template void
policy_dispatch<chunk_policy_head256, chunk_prefill_args_t<false>>(
    sycl::queue&,
    c10::ScalarType,
    c10::ScalarType,
    const chunk_prefill_args_t<false>&);

}  // namespace vllm::xpu::attn