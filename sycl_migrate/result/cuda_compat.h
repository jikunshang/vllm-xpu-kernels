#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#pragma once

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#endif

#if defined(USE_ROCM) && defined(__GFX9__)
  #define WARP_SIZE 64
#else
  #define WARP_SIZE 32
#endif

#ifndef USE_ROCM
  /*
  DPCT1098:65: The '*' expression is used instead of the __ldg call. These two
  expressions do not provide the exact same functionality. Check the generated
  code for potential precision and/or performance issues.
  */
  /*
  DPCT1064:66: Migrated __ldg call is used in a macro/template definition and
  may not be valid for all macro/template uses. Adjust the code.
  */
  #define VLLM_LDG(arg) *(sin_ptr + x_index / 2)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width)                    \
    dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane_mask, \
                                   width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
  /*
  DPCT1121:6: Make sure that the "sum" which is used in the SYCL group
  function/algorithm is initialized.
  */
  #define VLLM_SHFL_SYNC(var, src_lane) \
    dpct::select_from_sub_group(item_ct1.get_sub_group(), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
  #define VLLM_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
  /*
  DPCT1027:27: The call to cudaFuncSetAttribute was replaced with 0 because SYCL
  currently does not support corresponding setting.
  */
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) 0
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif
