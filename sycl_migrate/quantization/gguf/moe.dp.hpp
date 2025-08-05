#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
   based on ./vllm/model_executor/layers/fused_moe/fused_moe.py */
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
/*
DPCT1110:119: The total declared local variable size in device function moe_q
exceeds 128 bytes and may cause high register pressure. Consult with your
hardware vendor to find the total register size available and adjust the code,
or use smaller sub-group size to avoid high register pressure.
*/
static __dpct_inline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k, const sycl::nd_item<3>& item_ct1,
    int* tile_y_qs, sycl::half2* tile_y_ds) {
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int ncols_dst = ncols_y * top_k;

  const auto row_dst_0 = item_ct1.get_group(2) * mmq_y;
  const int& row_x_0 = row_dst_0;

  const auto col_dst_0 = item_ct1.get_group(1) * mmq_x;

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] =
        sorted_token_ids[col_dst_0 + item_ct1.get_local_id(1) + i];
  }

  const int exp_idx = expert_ids[item_ct1.get_group(1)];
  if (exp_idx > 255 || exp_idx < 0) return;
  if (item_ct1.get_group(1) * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  sycl::half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
               nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
               blocks_per_row_x);

    const int n_per_r = ((qk * blocks_per_warp) / qr);
#pragma unroll
    for (int ir = 0; ir < qr && ib0 * qk + ir * n_per_r < ncols_x; ++ir) {
      const auto kqs = ir * WARP_SIZE_GGUF + item_ct1.get_local_id(2);
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = token_offs[i / nwarps] / top_k;
        const int block_x = ib0 * (qk / QK8_1) + kbxd;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
          const int index_y = (item_ct1.get_local_id(1) + i) * WARP_SIZE_GGUF +
                              kqs % WARP_SIZE_GGUF;
          tile_y_qs[index_y] = get_int_from_int8_aligned(
              by0->qs, item_ct1.get_local_id(2) % QI8_1);
        }
      }

      if (item_ct1.get_local_id(2) < n_per_r / QK8_1) {
        const auto kby = item_ct1.get_local_id(2) % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = token_offs[item_ct1.get_local_id(1)] / top_k;
        const int block_x =
            ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const sycl::half2* dsi_src =
              &y[col_y_eff * blocks_per_col_y + block_x].ds;
          sycl::half2* dsi_dst =
              &tile_y_ds[item_ct1.get_local_id(1) * (WARP_SIZE_GGUF / QI8_1) +
                         kby];

          if (need_sum) {
            *dsi_dst = *dsi_src;
          } else {
            float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = (*dsi_src)[0];
          }
        }
      }
      /*
      DPCT1118:120: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:342: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr;
           k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs,
                        tile_y_ds, item_ct1.get_local_id(2) + i,
                        item_ct1.get_local_id(1) + j, k);
          }
        }
      }
      /*
      DPCT1118:121: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:343: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_0 8
  #define MOE_Y_Q4_0 128
  #define NWARPS_Q4_0 8
#else
  #define MOE_X_Q4_0 4
  #define MOE_Y_Q4_0 32
  #define NWARPS_Q4_0 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_q4_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q4_0;
  const int mmq_y = MOE_Y_Q4_0;
  const int nwarps = NWARPS_Q4_0;

  moe_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q4_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  int mmq_x = MOE_X_Q4_0;
  int mmq_y = MOE_Y_Q4_0;
  int nwarps = NWARPS_Q4_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:122: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:123: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_1 8
  #define MOE_Y_Q4_1 128
  #define NWARPS_Q4_1 8
#else
  #define MOE_X_Q4_1 4
  #define MOE_Y_Q4_1 32
  #define NWARPS_Q4_1 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_1, 2)
#endif
    moe_q4_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q4_1;
  const int mmq_y = MOE_Y_Q4_1;
  const int nwarps = NWARPS_Q4_1;

  moe_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_1<mmq_y>, load_tiles_q4_1<mmq_y, nwarps, need_check>,
        VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q4_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  int mmq_x = MOE_X_Q4_1;
  int mmq_y = MOE_Y_Q4_1;
  int nwarps = NWARPS_Q4_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:124: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_1<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:125: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_1<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_0 8
  #define MOE_Y_Q5_0 128
  #define NWARPS_Q5_0 8
#else
  #define MOE_X_Q5_0 4
  #define MOE_Y_Q5_0 32
  #define NWARPS_Q5_0 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_0, 2)
#endif
    moe_q5_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  moe_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_0<mmq_y>, load_tiles_q5_0<mmq_y, nwarps, need_check>,
        VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q5_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:126: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:127: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_1 8
  #define MOE_Y_Q5_1 128
  #define NWARPS_Q5_1 8
#else
  #define MOE_X_Q5_1 4
  #define MOE_Y_Q5_1 32
  #define NWARPS_Q5_1 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_1, 2)
#endif
    moe_q5_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  moe_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_1<mmq_y>, load_tiles_q5_1<mmq_y, nwarps, need_check>,
        VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q5_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:128: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_1<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:129: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_1<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q8_0 8
  #define MOE_Y_Q8_0 128
  #define NWARPS_Q8_0 8
#else
  #define MOE_X_Q8_0 4
  #define MOE_Y_Q8_0 32
  #define NWARPS_Q8_0 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q8_0, 2)
#endif
    moe_q8_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  moe_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q8_0<mmq_y>, load_tiles_q8_0<mmq_y, nwarps, need_check>,
        VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q8_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:130: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q8_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:131: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q8_0<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q2_K 8
  #define MOE_Y_Q2_K 128
  #define NWARPS_Q2_K 8
#else
  #define MOE_X_Q2_K 4
  #define MOE_Y_Q2_K 32
  #define NWARPS_Q2_K 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q2_K, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  moe_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q2_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q2_K;
  const int mmq_y = MOE_Y_Q2_K;
  const int nwarps = NWARPS_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:132: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q2_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:133: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q2_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q3_K 8
  #define MOE_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MOE_X_Q3_K 4
  #define MOE_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {

  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}
template <typename scalar_t>
static void ggml_moe_q3_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:134: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q3_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:135: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q3_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_K 8
  #define MOE_Y_Q4_K 128
  #define NWARPS_Q4_K 8
#else
  #define MOE_X_Q4_K 4
  #define MOE_Y_Q4_K 32
  #define NWARPS_Q4_K 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_K, 2)
#endif
    moe_q4_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  moe_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_K<mmq_y>, load_tiles_q4_K<mmq_y, nwarps, need_check>,
        VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:136: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:137: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q4_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_K 8
  #define MOE_Y_Q5_K 128
  #define NWARPS_Q5_K 8
#else
  #define MOE_X_Q5_K 4
  #define MOE_Y_Q5_K 32
  #define NWARPS_Q5_K 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_K, 2)
#endif
    moe_q5_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  moe_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_K<mmq_y>, load_tiles_q5_K<mmq_y, nwarps, need_check>,
        VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q5_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:138: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:139: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q5_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q6_K 8
  #define MOE_Y_Q6_K 128
  #define NWARPS_Q6_K 8
#else
  #define MOE_X_Q6_K 4
  #define MOE_Y_Q6_K 32
  #define NWARPS_Q6_K 4
#endif

template <typename scalar_t, bool need_check>
static void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q6_K, 2)
#endif
    moe_q6_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
             sycl::half2 *tile_y_ds) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  moe_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q6_K<mmq_y>, load_tiles_q6_K<mmq_y, nwarps, need_check>,
        VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k,
      item_ct1, tile_y_qs, tile_y_ds);
}

template <typename scalar_t>
static void ggml_moe_q6_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, dpct::queue_ptr stream) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dpct::dim3 block_nums(block_num_x, block_num_y, 1);
  const dpct::dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    /*
    DPCT1049:140: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q6_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  } else {
    constexpr bool need_check = true;
    /*
    DPCT1049:141: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF), cgh);
        sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
            sycl::range<1>(mmq_x * WARP_SIZE_GGUF / QI8_1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
              moe_q6_K<scalar_t, need_check>(
                  w, inp, dst, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, exp_stride, ncols_x, nrows_x, ncols_y,
                  nrows_y, nrows_dst, top_k, item_ct1,
                  tile_y_qs_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get(),
                  tile_y_ds_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
  }
}
