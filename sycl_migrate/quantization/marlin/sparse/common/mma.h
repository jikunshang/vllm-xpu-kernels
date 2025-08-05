#define DPCT_COMPAT_RT_VERSION 12080
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

// On CUDA earlier than 12.5, the ordered_metadata version of this instruction
// is not supported. On later versions of CUDA the version without ordered
// metadata results in the following warning:
//  | Advisory: Modifier ‘.sp::ordered_metadata’ should be used on instruction
//  | ‘mma’ instead of modifier ‘.sp’ as it is expected to have substantially
//  | reduced performance on some future architectures
#if defined DPCT_COMPAT_RT_VERSION && DPCT_COMPAT_RT_VERSION >= 12050
  #define MMA_SP_INST \
    "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
#else
  #define MMA_SP_INST "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
#endif

// m16n8k32 sparse tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
inline void mma_sp(const FragB& a_frag0, const FragB& a_frag1,
                              const FragA& frag_b, FragC& frag_c, FragM& frag_m,
                              const int psel) {
  const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a_frag0);
  const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a_frag1);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* e = reinterpret_cast<const uint32_t*>(&frag_m);

  float* c = reinterpret_cast<float*>(&frag_c);
  if (psel == 0) {
    /*
    DPCT1053:43: Migration of device assembly code is not supported.
    */
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    /*
    DPCT1053:44: Migration of device assembly code is not supported.
    */
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
  } else {
    /*
    DPCT1053:45: Migration of device assembly code is not supported.
    */
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[0]),
                   "r"(b[2]), "r"(b[4]), "r"(b[6]), "f"(c[0]), "f"(c[1]),
                   "f"(c[2]), "f"(c[3]), "r"(e[0]));
    /*
    DPCT1053:46: Migration of device assembly code is not supported.
    */
    asm volatile(MMA_SP_INST
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x1;\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a0[0]), "r"(a1[0]), "r"(a0[1]), "r"(a1[1]), "r"(b[1]),
                   "r"(b[3]), "r"(b[5]), "r"(b[7]), "f"(c[4]), "f"(c[5]),
                   "f"(c[6]), "f"(c[7]), "r"(e[0]));
  }
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
inline int lop3(int a, int b, int c) {
  int res;
  res = dpct::ternary_logic_op(a, b, c, lut);
  return res;
}

__dpct_inline__ sycl::uint2 to_half4(float c0, float c1, float c2, float c3) {
  sycl::uint2 r;
  {
    sycl::half a, b, c, d;
    a = sycl::vec<float, 1>(c0)
            .template convert<sycl::half, sycl::rounding_mode::rte>()
            .x();
    b = sycl::vec<float, 1>(c1)
            .template convert<sycl::half, sycl::rounding_mode::rte>()
            .x();
    c = sycl::vec<float, 1>(c2)
            .template convert<sycl::half, sycl::rounding_mode::rte>()
            .x();
    d = sycl::vec<float, 1>(c3)
            .template convert<sycl::half, sycl::rounding_mode::rte>()
            .x();
    r.x() =
        sycl::vec<uint16_t, 2>({a, b}).template as<sycl::vec<uint32_t, 1>>()[0];
    r.y() =
        sycl::vec<uint16_t, 2>({c, d}).template as<sycl::vec<uint32_t, 1>>()[0];
  }
  return r;
}

// Constructs destination register by taking bytes from 2 sources (based on
// mask)
template <int start_byte, int mask>
inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  /*
  DPCT1053:47: Migration of device assembly code is not supported.
  */
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline FragB dequant_4bit(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  FragB frag_b;
  frag_b[0] = *reinterpret_cast<sycl::half2*>(&lo) -
              *reinterpret_cast<const sycl::half2*>(&SUB);
  frag_b[1] = sycl::fma(*reinterpret_cast<sycl::half2*>(&hi),
                        *reinterpret_cast<const sycl::half2*>(&MUL),
                        *reinterpret_cast<const sycl::half2*>(&ADD));
  return frag_b;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline FragB dequant_8bit(int q) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;

  FragB frag_b;
  frag_b[0] = *reinterpret_cast<sycl::half2*>(&lo) -
              *reinterpret_cast<const sycl::half2*>(&I8s_TO_F16s_MAGIC_NUM);
  frag_b[1] = *reinterpret_cast<sycl::half2*>(&hi) -
              *reinterpret_cast<const sycl::half2*>(&I8s_TO_F16s_MAGIC_NUM);
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  sycl::half2 s = sycl::half2(reinterpret_cast<sycl::half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

inline void scale_floats(float* c0, float* c1, float* c2, float* c3,
                                    FragS& s0, float* c4, float* c5, float* c6,
                                    float* c7, FragS& s1) {
  *c0 = __fmul_rn(*c0, __half2float(s0[0].x));
  *c1 = __fmul_rn(*c1, __half2float(s0[0].y));
  *c2 = __fmul_rn(*c2, __half2float(s0[1].x));
  *c3 = __fmul_rn(*c3, __half2float(s0[1].y));

  *c4 = __fmul_rn(*c4, __half2float(s1[0].x));
  *c5 = __fmul_rn(*c5, __half2float(s1[0].y));
  *c6 = __fmul_rn(*c6, __half2float(s1[1].x));
  *c7 = __fmul_rn(*c7, __half2float(s1[1].y));
}

}  // namespace marlin_24
