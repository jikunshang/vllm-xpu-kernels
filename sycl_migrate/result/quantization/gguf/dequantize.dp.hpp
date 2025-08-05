#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// copied and adapted from
// https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/convert.cu
// Dequant functions
static __dpct_inline__ void dequantize_q4_0(const void* vx, const int ib,
                                            const int iqs, dfloat2& v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = sycl::vec<int, 1>(vui & 0xF)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    v.y() = sycl::vec<int, 1>(vui >> 4)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    v = v - sycl::float2(8.0f, 8.0f)
                .convert<sycl::half, sycl::rounding_mode::rte>();
    v = v * {d, d};
}

static __dpct_inline__ void dequantize_q4_1(const void* vx, const int ib,
                                            const int iqs, dfloat2& v) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    const int vui = x[ib].qs[iqs];

    v.x() = sycl::vec<int, 1>(vui & 0xF)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    v.y() = sycl::vec<int, 1>(vui >> 4)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    v = v * {d, d};
    v = v + {m, m};
}

static __dpct_inline__ void dequantize_q5_0(const void* vx, const int ib,
                                            const int iqs, dfloat2& v) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = sycl::vec<int, 1>((x[ib].qs[iqs] & 0xf) | xh_0)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    v.y() = sycl::vec<int, 1>((x[ib].qs[iqs] >> 4) | xh_1)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    v = v - sycl::float2(16.0f, 16.0f)
                .convert<sycl::half, sycl::rounding_mode::rte>();
    v = v * {d, d};
}

static __dpct_inline__ void dequantize_q5_1(const void* vx, const int ib,
                                            const int iqs, dfloat2& v) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = x[ib].dm[0];
    const dfloat m = x[ib].dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = sycl::vec<int, 1>((x[ib].qs[iqs] & 0xf) | xh_0)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    v.y() = sycl::vec<int, 1>((x[ib].qs[iqs] >> 4) | xh_1)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    v = v * {d, d};
    v = v + {m, m};
}

static __dpct_inline__ void dequantize_q8_0(const void* vx, const int ib,
                                            const int iqs, dfloat2& v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x() = sycl::vec<int, 1>(x[ib].qs[iqs + 0])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    v.y() = sycl::vec<int, 1>(x[ib].qs[iqs + 1])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    v = v * {d, d};
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k,
                             const sycl::nd_item<3> &item_ct1) {
    const int i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = convert_from_half<dst_t>(v.x());
    y[iybs + iqs + y_offset] = convert_from_half<dst_t>(v.y());
}

template<typename dst_t>
static void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const auto i = item_ct1.get_group(2);
    const block_q2_K * x = (const block_q2_K *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    sycl::half dall = x[i].dm[0];
    sycl::half dmin = x[i].dm[1];
    y[l + 0] = convert_from_half<dst_t>(
        dall * sycl::vec<int, 1>((x[i].scales[is + 0] & 0xF) * ((q >> 0) & 3))
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        dmin * sycl::vec<int, 1>(x[i].scales[is + 0] >> 4)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[l + 32] = convert_from_half<dst_t>(
        dall * sycl::vec<int, 1>((x[i].scales[is + 2] & 0xF) * ((q >> 2) & 3))
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        dmin * sycl::vec<int, 1>(x[i].scales[is + 2] >> 4)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[l + 64] = convert_from_half<dst_t>(
        dall * sycl::vec<int, 1>((x[i].scales[is + 4] & 0xF) * ((q >> 4) & 3))
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        dmin * sycl::vec<int, 1>(x[i].scales[is + 4] >> 4)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[l + 96] = convert_from_half<dst_t>(
        dall * sycl::vec<int, 1>((x[i].scales[is + 6] & 0xF) * ((q >> 6) & 3))
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        dmin * sycl::vec<int, 1>(x[i].scales[is + 6] >> 4)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

template<typename dst_t>
static void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const auto i = item_ct1.get_group(2);
    const block_q3_K * x = (const block_q3_K *) vx;

    const auto r = item_ct1.get_local_id(2) / 4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    sycl::half d_all = x[i].d;
    sycl::half dl =
        d_all * sycl::vec<int, 1>(us - 32)
                    .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = convert_from_half<dst_t>(
            dl * sycl::vec<int, 1>((int8_t)((q[l] >> shift) & 3) -
                                   ((hm[l] & m) ? 0 : 4))
                     .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    }
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const auto i = item_ct1.get_group(2);

    // assume 32 threads
    const auto tid = item_ct1.get_local_id(2);
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const sycl::half dall = x[i].dm[0];
    const sycl::half dmin = x[i].dm[1];

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const sycl::half d1 =
        dall * sycl::vec<int, 1>(sc)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    const sycl::half m1 =
        dmin *
        sycl::vec<int, 1>(m).convert<sycl::half, sycl::rounding_mode::rte>()[0];
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const sycl::half d2 =
        dall * sycl::vec<int, 1>(sc)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    const sycl::half m2 =
        dmin *
        sycl::vec<int, 1>(m).convert<sycl::half, sycl::rounding_mode::rte>()[0];
    for (int l = 0; l < n; ++l) {
        y[l + 0] = convert_from_half<dst_t>(
            d1 * sycl::vec<int, 1>(q[l] & 0xF)
                     .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
            m1);
        y[l + 32] = convert_from_half<dst_t>(
            d2 * sycl::vec<int, 1>(q[l] >> 4)
                     .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
            m2);
    }
}

template<typename dst_t>
static void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const auto i = item_ct1.get_group(2);

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = item_ct1.get_local_id(2);
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const sycl::half dall = x[i].dm[0];
    const sycl::half dmin = x[i].dm[1];

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const sycl::half d1 =
        dall * sycl::vec<int, 1>(sc)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        const sycl::half m1 =
            dmin * sycl::vec<int, 1>(m)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const sycl::half d2 =
        dall * sycl::vec<int, 1>(sc)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0];
        const sycl::half m2 =
            dmin * sycl::vec<int, 1>(m)
                       .convert<sycl::half, sycl::rounding_mode::rte>()[0];

    uint8_t   hm  = 1 << (2*il);
    y[0] = convert_from_half<dst_t>(
        d1 * sycl::vec<int, 1>((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))
                 .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        m1);
    y[1] = convert_from_half<dst_t>(
        d1 * sycl::vec<int, 1>((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))
                 .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        m1);
    hm <<= 1;
    y[32] = convert_from_half<dst_t>(
        d2 * sycl::vec<int, 1>((ql[0] >> 4) + (qh[0] & hm ? 16 : 0))
                 .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        m2);
    y[33] = convert_from_half<dst_t>(
        d2 * sycl::vec<int, 1>((ql[1] >> 4) + (qh[1] & hm ? 16 : 0))
                 .convert<sycl::half, sycl::rounding_mode::rte>()[0] -
        m2);
}

template<typename dst_t>
static void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  const sycl::nd_item<3> &item_ct1) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const auto i = item_ct1.get_group(2);

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = item_ct1.get_local_id(2);
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const sycl::half d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[0] = convert_from_half<dst_t>(
        d * sycl::vec<int, 1>(
                sc[0] * ((int8_t)((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32))
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[32] = convert_from_half<dst_t>(
        d *
        sycl::vec<int, 1>(
            sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32))
            .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[64] = convert_from_half<dst_t>(
        d * sycl::vec<int, 1>(
                sc[4] * ((int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32))
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
    y[96] = convert_from_half<dst_t>(
        d * sycl::vec<int, 1>(
                sc[6] * ((int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32))
                .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}

template<typename dst_t>
static void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                     const sycl::nd_item<3> &item_ct1,
                                     const uint64_t *iq2xxs_grid,
                                     const uint8_t *ksigns_iq2xs,
                                     const uint8_t *kmask_iq2xs) {
    const auto i = item_ct1.get_group(2);
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                    const sycl::nd_item<3> &item_ct1,
                                    const uint64_t *iq2xs_grid,
                                    const uint8_t *ksigns_iq2xs,
                                    const uint8_t *kmask_iq2xs) {
    const auto i = item_ct1.get_group(2);
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);

}

template<typename dst_t>
static void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                   const sycl::nd_item<3> &item_ct1,
                                   const uint64_t *iq2s_grid,
                                   const uint8_t *kmask_iq2xs) {
    const auto i = item_ct1.get_group(2);
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                     const sycl::nd_item<3> &item_ct1,
                                     const uint32_t *iq3xxs_grid,
                                     const uint8_t *ksigns_iq2xs,
                                     const uint8_t *kmask_iq2xs) {
    const auto i = item_ct1.get_group(2);
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                   const sycl::nd_item<3> &item_ct1,
                                   const uint32_t *iq3xs_grid,
                                   const uint8_t *kmask_iq2xs) {
    const auto i = item_ct1.get_group(2);
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xs_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3xs_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (0.5f + ((x[i].scales[ib / 2] >> 4 * (ib % 2)) & 0xf)) *
                    0.5f;
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                   const sycl::nd_item<3> &item_ct1,
                                   const uint64_t *iq1s_grid_gpu) {
    const int64_t i = item_ct1.get_group(2);
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (2 * ((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                   const sycl::nd_item<3> &item_ct1,
                                   const uint64_t *iq1s_grid_gpu) {
    const int64_t i = item_ct1.get_group(2);
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = sycl::vec<sycl::half, 1>(scale.f16)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                    const sycl::nd_item<3> &item_ct1,
                                    const int8_t *kvalues_iq4nl) {
    const auto i = item_ct1.get_group(2);
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = sycl::vec<sycl::half, 1>(x[ib].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0];
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}

template<typename dst_t>
static void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                    const sycl::nd_item<3> &item_ct1,
                                    const int8_t *kvalues_iq4nl) {
    const auto i = item_ct1.get_group(2);
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const auto tid = item_ct1.get_local_id(2);
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = sycl::vec<sycl::half, 1>(x[i].d)
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    ((((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                      (((x[i].scales_h >> 2 * ib) & 3) << 4)) -
                     32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void* __restrict__ vx,
                                  dst_t* __restrict__ y, const int k,
                                  dpct::queue_ptr stream) {
    const int num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, CUDA_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, CUDA_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_cuda(const void* vx, dst_t* y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q3_K_cuda(const void* vx, dst_t* y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_cuda(const void* vx, dst_t* y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q5_K_cuda(const void* vx, dst_t* y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q5_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q6_K_cuda(const void* vx, dst_t* y, const int k,
                                     dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void* vx, dst_t* y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint64_t, 1> iq2xxs_grid;
        extern dpct::global_memory<uint8_t, 1> ksigns_iq2xs;
        extern dpct::global_memory<uint8_t, 1> kmask_iq2xs;

        iq2xxs_grid.init(*stream);
        ksigns_iq2xs.init(*stream);
        kmask_iq2xs.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
            auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
            auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid_ptr_ct1,
                                     ksigns_iq2xs_ptr_ct1, kmask_iq2xs_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void* vx, dst_t* y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint64_t, 1> iq2xs_grid;
        extern dpct::global_memory<uint8_t, 1> ksigns_iq2xs;
        extern dpct::global_memory<uint8_t, 1> kmask_iq2xs;

        iq2xs_grid.init(*stream);
        ksigns_iq2xs.init(*stream);
        kmask_iq2xs.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
            auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
            auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid_ptr_ct1,
                                     ksigns_iq2xs_ptr_ct1, kmask_iq2xs_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_cuda(const void* vx, dst_t* y, const int k,
                                      dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint64_t, 1> iq2s_grid;
        extern dpct::global_memory<uint8_t, 1> kmask_iq2xs;

        iq2s_grid.init(*stream);
        kmask_iq2xs.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq2s_grid_ptr_ct1 = iq2s_grid.get_ptr();
            auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1,
                                                        iq2s_grid_ptr_ct1,
                                                        kmask_iq2xs_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void* vx, dst_t* y, const int k,
                                        dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint32_t, 1> iq3xxs_grid;
        extern dpct::global_memory<uint8_t, 1> ksigns_iq2xs;
        extern dpct::global_memory<uint8_t, 1> kmask_iq2xs;

        iq3xxs_grid.init(*stream);
        ksigns_iq2xs.init(*stream);
        kmask_iq2xs.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
            auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
            auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid_ptr_ct1,
                                     ksigns_iq2xs_ptr_ct1, kmask_iq2xs_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_cuda(const void* vx, dst_t* y, const int k,
                                      dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint32_t, 1> iq3xs_grid;
        extern dpct::global_memory<uint8_t, 1> kmask_iq2xs;

        iq3xs_grid.init(*stream);
        kmask_iq2xs.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq3xs_grid_ptr_ct1 = iq3xs_grid.get_ptr();
            auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(vx, y, item_ct1,
                                                        iq3xs_grid_ptr_ct1,
                                                        kmask_iq2xs_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_s_cuda(const void* vx, dst_t* y, const int k,
                                      dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint64_t, 1> iq1s_grid_gpu;

        iq1s_grid_gpu.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(vx, y, item_ct1,
                                                        iq1s_grid_gpu_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_cuda(const void* vx, dst_t* y, const int k,
                                      dpct::queue_ptr stream) {
    const int nb = k / QK_K;
    {
        extern dpct::global_memory<uint64_t, 1> iq1s_grid_gpu;

        iq1s_grid_gpu.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto iq1s_grid_gpu_ptr_ct1 = iq1s_grid_gpu.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(vx, y, item_ct1,
                                                        iq1s_grid_gpu_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void* vx, dst_t* y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    {
        extern dpct::global_memory<int8_t, 1> kvalues_iq4nl;

        kvalues_iq4nl.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq4_nl(vx, y, item_ct1,
                                                         kvalues_iq4nl_ptr_ct1);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void* vx, dst_t* y, const int k,
                                       dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    {
        extern dpct::global_memory<int8_t, 1> kvalues_iq4nl;

        kvalues_iq4nl.init(*stream);

        stream->submit([&](sycl::handler& cgh) {
            auto kvalues_iq4nl_ptr_ct1 = kvalues_iq4nl.get_ptr();

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq4_xs(vx, y, item_ct1,
                                                         kvalues_iq4nl_ptr_ct1);
                             });
        });
    }
}

template<typename dst_t>
static to_cuda_ggml_t<dst_t> ggml_get_to_cuda(int64_t type) {
    switch (type) {
        case 2:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case 3:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case 6:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case 7:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case 8:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case 10:
            return dequantize_row_q2_K_cuda;
        case 11:
            return dequantize_row_q3_K_cuda;
        case 12:
            return dequantize_row_q4_K_cuda;
        case 13:
            return dequantize_row_q5_K_cuda;
        case 14:
            return dequantize_row_q6_K_cuda;
        case 16:
            return dequantize_row_iq2_xxs_cuda;
        case 17:
            return dequantize_row_iq2_xs_cuda;
        case 18:
            return dequantize_row_iq3_xxs_cuda;
        case 19:
            return dequantize_row_iq1_s_cuda;
        case 20:
            return dequantize_row_iq4_nl_cuda;
        case 21:
            return dequantize_row_iq3_s_cuda;
        case 22:
            return dequantize_row_iq2_s_cuda;
        case 23:
            return dequantize_row_iq4_xs_cuda;
        case 29:
            return dequantize_row_iq1_m_cuda;
        default:
            return nullptr;
    }
}
