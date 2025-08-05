/*
 * Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All
 * Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "base.h"

namespace marlin_24 {
// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
inline void cp_async4_pred_zfill(void* smem_ptr,
                                            const void* glob_ptr,
                                            bool pred = true,
                                            const bool zfill = false) {
  const int BYTES = 16;
  int src_in_bytes = (zfill ? 0 : BYTES);
  auto smem = smem_ptr;
  /*
  DPCT1053:35: Migration of device assembly code is not supported.
  */
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES), "r"(src_in_bytes));
}

inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  auto smem = smem_ptr;
  /*
  DPCT1053:36: Migration of device assembly code is not supported.
  */
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy
inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  auto smem = smem_ptr;
  /*
  DPCT1053:37: Migration of device assembly code is not supported.
  */
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
inline void cp_async_fence() {
  /*
  DPCT1053:38: Migration of device assembly code is not supported.
  */
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
inline void cp_async_wait() {
  /*
  DPCT1053:39: Migration of device assembly code is not supported.
  */
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  auto smem = smem_ptr;
  /*
  DPCT1053:40: Migration of device assembly code is not supported.
  */
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

inline void ldsm4_m(FragM& frag_m, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_m);
  auto smem = smem_ptr;
  /*
  DPCT1053:41: Migration of device assembly code is not supported.
  */
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1])
               : "r"(smem));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
inline void ldsm4_t(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  auto smem = smem_ptr;
  /*
  DPCT1053:42: Migration of device assembly code is not supported.
  */
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
inline void barrier_acquire(int* lock, int count,
                            const sycl::nd_item<3> &item_ct1) {
  if (item_ct1.get_local_id(2) == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      state = *((uint32_t*)(uintptr_t)lock);
    while (state != count);
  }
  /*
  DPCT1065:315: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
}

// Release barrier and increment visitation count.
inline void barrier_release(int* lock, const sycl::nd_item<3> &item_ct1, bool reset = false) {
  /*
  DPCT1065:316: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();
  if (item_ct1.get_local_id(2) == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    *lock = sycl::reduce_over_group(item_ct1.get_group(), val, sycl::plus<>());
  }
}
}  // namespace marlin_24
