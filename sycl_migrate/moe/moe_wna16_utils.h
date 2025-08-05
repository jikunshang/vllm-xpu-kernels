
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

template <typename scalar_t>
class ScalarType {};

template <>
class ScalarType<sycl::half> {
 public:
  using scalar_t = sycl::half;
  using scalar_t2 = sycl::half2;

  static float inline num2float(const sycl::half x) {
    return sycl::vec<sycl::half, 1>(x)
        .convert<float, sycl::rounding_mode::automatic>()[0];
  }

  static sycl::half2 inline num2num2(const sycl::half x) {
    return sycl::half2(x);
  }

  static sycl::half2 inline nums2num2(const sycl::half x1,
                                      const sycl::half x2) {
    return sycl::half2(x1, x2);
  }

  static sycl::half inline float2num(const float x) {
    return sycl::vec<float, 1>(x)
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
  }

  static sycl::half inline int2num(const float x) {
    return sycl::vec<int, 1>(x)
        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
  }

  static sycl::float2 inline num22float2(const sycl::half2 x) {
    return x.convert<float, sycl::rounding_mode::automatic>();
  }

  static sycl::half2 inline float22num2(const sycl::float2 x) {
    return x.convert<sycl::half, sycl::rounding_mode::rte>();
  }
};

template <>
class ScalarType<sycl::ext::oneapi::bfloat16> {
 public:
  using scalar_t = sycl::ext::oneapi::bfloat16;
  using scalar_t2 = sycl::marray<sycl::ext::oneapi::bfloat16, 2>;

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
  static float inline num2float(const sycl::ext::oneapi::bfloat16 x) {
    return static_cast<float>(x);
  }

  static sycl::marray<sycl::ext::oneapi::bfloat16, 2> inline num2num2(
      const sycl::ext::oneapi::bfloat16 x) {
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x, x);
  }

  static sycl::marray<sycl::ext::oneapi::bfloat16, 2> inline nums2num2(
      const sycl::ext::oneapi::bfloat16 x1,
      const sycl::ext::oneapi::bfloat16 x2) {
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x1, x2);
  }

  static sycl::ext::oneapi::bfloat16 inline float2num(const float x) {
    return sycl::ext::oneapi::bfloat16(x);
  }

  static sycl::ext::oneapi::bfloat16 inline int2num(const float x) {
    /*
    DPCT1007:291: Migration of __int2bfloat16_rn is not supported.
    */
    return __int2bfloat16_rn(x);
  }

  static sycl::float2 inline num22float2(
      const sycl::marray<sycl::ext::oneapi::bfloat16, 2> x) {
    return sycl::float2(x[0], x[1]);
  }

  static sycl::marray<sycl::ext::oneapi::bfloat16, 2> inline float22num2(
      const sycl::float2 x) {
    return sycl::marray<sycl::ext::oneapi::bfloat16, 2>(x[0], x[1]);
  }
#endif
};

template <int lut>
inline int lop3(int a, int b, int c) {
  int res;
  res = dpct::ternary_logic_op(a, b, c, lut);
  return res;
}

template <int start_byte, int mask>
inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  /*
  DPCT1053:20: Migration of device assembly code is not supported.
  */
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

template <typename scalar_t2, int bit>
inline void dequant(int q, scalar_t2* res) {}

template <>
inline void dequant<half2, 4>(int q, sycl::half2* res) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  q >>= 8;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  res[0] = *reinterpret_cast<sycl::half2*>(&lo0) -
           *reinterpret_cast<const sycl::half2*>(&SUB);
  res[1] = sycl::fma(*reinterpret_cast<sycl::half2*>(&hi0),
                     *reinterpret_cast<const sycl::half2*>(&MUL),
                     *reinterpret_cast<const sycl::half2*>(&ADD));
  res[2] = *reinterpret_cast<sycl::half2*>(&lo1) -
           *reinterpret_cast<const sycl::half2*>(&SUB);
  res[3] = sycl::fma(*reinterpret_cast<sycl::half2*>(&hi1),
                     *reinterpret_cast<const sycl::half2*>(&MUL),
                     *reinterpret_cast<const sycl::half2*>(&ADD));
}

template <>
inline void dequant<half2, 8>(int q, sycl::half2* res) {
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  res[0] = *reinterpret_cast<sycl::half2*>(&lo) -
           *reinterpret_cast<const sycl::half2*>(&I8s_TO_F16s_MAGIC_NUM);
  res[1] = *reinterpret_cast<sycl::half2*>(&hi) -
           *reinterpret_cast<const sycl::half2*>(&I8s_TO_F16s_MAGIC_NUM);
}

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
template <>
inline void dequant<nv_bfloat162, 4>(
    int q, sycl::marray<sycl::ext::oneapi::bfloat16, 2>* res) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;

  res[0] =
      *reinterpret_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&lo0) *
          *reinterpret_cast<
              const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&MUL) +
      *reinterpret_cast<const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(
          &ADD);
  res[1] =
      *reinterpret_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&hi0) *
          *reinterpret_cast<
              const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&MUL) +
      *reinterpret_cast<const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(
          &ADD);
  res[2] =
      *reinterpret_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&lo1) *
          *reinterpret_cast<
              const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&MUL) +
      *reinterpret_cast<const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(
          &ADD);
  res[3] =
      *reinterpret_cast<sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&hi1) *
          *reinterpret_cast<
              const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(&MUL) +
      *reinterpret_cast<const sycl::marray<sycl::ext::oneapi::bfloat16, 2>*>(
          &ADD);
}

template <>
inline void dequant<nv_bfloat162, 8>(
    int q, sycl::marray<sycl::ext::oneapi::bfloat16, 2>* res) {
  float fp32_intermediates[4];
  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = dpct::byte_level_permute(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = dpct::byte_level_permute(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = dpct::byte_level_permute(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = dpct::byte_level_permute(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(res);
  bf16_result_ptr[0] = dpct::byte_level_permute(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = dpct::byte_level_permute(
      fp32_intermediates_casted[2], fp32_intermediates_casted[3], 0x7632);
}
#endif
