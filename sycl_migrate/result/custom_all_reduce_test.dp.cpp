/**
 * This is a standalone test for custom allreduce.
 * To compile, make sure you have MPI and NCCL installed in your system.
 * export MPI_HOME=XXX
 * nvcc -O2 -arch=native -std=c++17 custom_all_reduce_test.cu -o
 * custom_all_reduce_test -lnccl -I${MPI_HOME}/include -lmpi
 *
 * Warning: this C++ test is not designed to be very readable and was used
 * during the rapid prototyping process.
 *
 * To run:
 * mpirun --allow-run-as-root -np 8 ./custom_all_reduce_test
 */
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <stdio.h>
#include <stdlib.h>

#include <limits>
#include <vector>

#include "custom_all_reduce.dp.hpp"
#include "mpi.h"
#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 nv_bfloat16;
  #include "rccl/rccl.h"
  #include "custom_all_reduce_hip.cuh"
#else
  #include "nccl.h"
  #include "custom_all_reduce.dp.hpp"
  #include <time.h>

  #include <cmath>

#endif

#define MPICHECK(cmd)                                                  \
  do {                                                                 \
    int e = cmd;                                                       \
    if (e != MPI_SUCCESS) {                                            \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#ifdef USE_ROCM
__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) {
    uint64_t start = wall_clock64();
    uint64_t cycles_elapsed;
    do {
      cycles_elapsed = wall_clock64() - start;
    } while (cycles_elapsed < 100);
  }
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}
#else
void dummy_kernel() {
  #if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 700
  /*
  DPCT1008:44: __nanosleep function is not defined in SYCL. This is a
  hardware-specific feature. Consult with your hardware vendor to find a
  replacement.
  */
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
  #else
  for (int i = 0; i < 100; i++) {
    long long int start = clock64();
    while (clock64() - start < 150000000);  // approximately 98.4ms on P40
  }
  #endif
}
#endif

template <typename T>
void set_data(T* data, int size, int myRank, const sycl::nd_item<3> &item_ct1) {
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
       idx < size;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    data[idx] = myRank * 0.11f;
  }
}

template <typename T>
void convert_data(const T* data1, const T* data2, double* fdata1,
                             double* fdata2, int size,
                             const sycl::nd_item<3> &item_ct1) {
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
       idx < size;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    fdata1[idx] = data1[idx];
    fdata2[idx] = data2[idx];
  }
}

/*
DPCT1032:45: A different random number generator is used. You may need to adjust
the code.
*/
void init_rand(
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>* state,
    int size, int nRanks, const sycl::nd_item<3>& item_ct1) {
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
       idx < size;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    for (int i = 0; i < nRanks; i++) {
      /*
      DPCT1105:46: The mcg59 random number generator is used. The subsequence
      argument "idx" is ignored. You need to verify the migration.
      */
      state[idx * nRanks + i] =
          dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>(
              i + 1, 0);
    }
  }
}

template <typename T>
/*
DPCT1032:47: A different random number generator is used. You may need to adjust
the code.
*/
void gen_data(
    dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>* state,
    T* data, double* ground_truth, int myRank, int nRanks, int size,
    const sycl::nd_item<3>& item_ct1) {
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
       idx < size;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range(2)) {
    double sum = 0.0;
    for (int i = 0; i < nRanks; i++) {
      double val =
          state[idx * nRanks + i]
              .generate<oneapi::mkl::rng::device::uniform<double>, 1>() *
          4;
      T hval = val;  // downcast first
      sum += static_cast<double>(hval);
      if (i == myRank) data[idx] = hval;
    }
    ground_truth[idx] = sum;
  }
}

template <typename T>
void run(int myRank, int nRanks, ncclComm_t& comm, int threads, int block_limit,
         int data_size, bool performance_test) try {
  T* result;
  dpct::queue_ptr stream;
  /*
  DPCT1025:48: The SYCL queue is created ignoring the flag and priority options.
  */
  CUDACHECK(
      DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue()));
  CUDACHECK(
      DPCT_CHECK_ERROR(result = (T*)sycl::malloc_device(
                           data_size * sizeof(T), dpct::get_in_order_queue())));
  CUDACHECK(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(result, 0, data_size * sizeof(T))
                                 .wait()));

  cudaIpcMemHandle_t self_data_handle;
  cudaIpcMemHandle_t data_handles[8];
  vllm::Signal* buffer;
  T* self_data_copy;
  /**
   * Allocate IPC buffer
   *
   * The first section is a temporary buffer for storing intermediate allreduce
   * results, if a particular algorithm requires it. The second section is for
   * the input to the allreduce. The actual API takes the input pointer as an
   * argument (that is, they can and usually should be allocated separately).
   * But since the input pointers and the temporary buffer all require IPC
   * registration, they are allocated and registered together in the test for
   * convenience.
   */
#ifdef USE_ROCM
  CUDACHECK(hipExtMallocWithFlags(
      (void**)&buffer, 2 * data_size * sizeof(T) + sizeof(vllm::Signal),
      hipDeviceMallocUncached));
#else
  CUDACHECK(
      cudaMalloc(&buffer, 2 * data_size * sizeof(T) + sizeof(vllm::Signal)));
#endif
  CUDACHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue()
          .memset(buffer, 0, 2 * data_size * sizeof(T) + sizeof(vllm::Signal))
          .wait()));
  CUDACHECK(
      DPCT_CHECK_ERROR(self_data_copy = (T*)sycl::malloc_device(
                           data_size * sizeof(T), dpct::get_in_order_queue())));
  /*
  DPCT1030:49: SYCL currently does not support inter-process communication (IPC)
  operations. You may need to rewrite the code.
  */
  CUDACHECK(cudaIpcGetMemHandle(&self_data_handle, buffer));

  MPICHECK(MPI_Allgather(&self_data_handle, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, data_handles, sizeof(cudaIpcMemHandle_t),
                         MPI_BYTE, MPI_COMM_WORLD));

  void* rank_data;
  size_t rank_data_sz = 16 * 1024 * 1024;
  CUDACHECK(DPCT_CHECK_ERROR(rank_data = (void*)sycl::malloc_device(
                                 rank_data_sz, dpct::get_in_order_queue())));
  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < nRanks; i++) {
    if (i == myRank)
      ipc_ptrs[i] = buffer;
    else
      /*
      DPCT1030:50: SYCL currently does not support inter-process communication
      (IPC) operations. You may need to rewrite the code.
      */
      CUDACHECK(cudaIpcOpenMemHandle((void**)&ipc_ptrs[i], data_handles[i],
                                     cudaIpcMemLazyEnablePeerAccess));
  }
  vllm::CustomAllreduce fa(ipc_ptrs, rank_data, rank_data_sz, myRank, nRanks);
  auto* self_data =
      reinterpret_cast<T*>(reinterpret_cast<char*>(buffer) +
                           sizeof(vllm::Signal) + data_size * sizeof(T));
  // hack buffer registration
  {
    void* data[8];
    for (int i = 0; i < nRanks; i++) {
      data[i] =
          ((char*)ipc_ptrs[i]) + sizeof(vllm::Signal) + data_size * sizeof(T);
    }
    fa.register_buffer(data);
  }

  double* ground_truth;
  CUDACHECK(cudaMallocHost(&ground_truth, data_size * sizeof(double)));
  /*
  DPCT1032:51: A different random number generator is used. You may need to
  adjust the code.
  */
  dpct::rng::device::rng_generator<oneapi::mkl::rng::device::mcg59<1>>* states;
  /*
  DPCT1032:52: A different random number generator is used. You may need to
  adjust the code.
  */
  CUDACHECK(cudaMalloc(&states, sizeof(dpct::rng::device::rng_generator<
                                       oneapi::mkl::rng::device::mcg59<1>>) *
                                    nRanks * data_size));
  /*
  DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
 stream->parallel_for(
     sycl::nd_range<3>(sycl::range<3>(1, 1, 108) * sycl::range<3>(1, 1, 1024),
                       sycl::range<3>(1, 1, 1024)),
     [=](sycl::nd_item<3> item_ct1) {
      init_rand(states, data_size, nRanks, item_ct1);
     });
  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
 {
  dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

  stream->parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 108) * sycl::range<3>(1, 1, 1024),
                        sycl::range<3>(1, 1, 1024)),
      [=](sycl::nd_item<3> item_ct1) {
       gen_data<T>(states, self_data, ground_truth, myRank, nRanks, data_size,
                   item_ct1);
      });
 }
  /*
  DPCT1124:53: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  CUDACHECK(DPCT_CHECK_ERROR(
      stream->memcpy(self_data_copy, self_data, data_size * sizeof(T))));
  dpct::event_ptr start, stop;
  CUDACHECK(DPCT_CHECK_ERROR(start = new sycl::event()));
  CUDACHECK(DPCT_CHECK_ERROR(stop = new sycl::event()));

  ncclDataType_t ncclDtype;
  if (std::is_same<T, sycl::half>::value) {
    ncclDtype = ncclFloat16;
  } else if (std::is_same<T, sycl::ext::oneapi::bfloat16>::value) {
    ncclDtype = ncclBfloat16;
  } else {
    ncclDtype = ncclFloat;
  }
  double *nccl_result, *my_result;
  CUDACHECK(cudaMallocHost(&nccl_result, data_size * sizeof(double)));
  CUDACHECK(cudaMallocHost(&my_result, data_size * sizeof(double)));
  if (performance_test) {
  stream->parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
       dummy_kernel();
      });
    constexpr int warmup_iters = 5;
    constexpr int num_iters = 100;
    // warmup
    for (int i = 0; i < warmup_iters; i++) {
      NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum,
                              comm, stream));
    }
    /*
    DPCT1024:54: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CUDACHECK(DPCT_CHECK_ERROR(*start = stream->ext_oneapi_submit_barrier()));
    for (int i = 0; i < num_iters; i++) {
      NCCLCHECK(ncclAllReduce(result, result, data_size, ncclDtype, ncclSum,
                              comm, stream));
    }
    /*
    DPCT1024:55: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CUDACHECK(DPCT_CHECK_ERROR(*stop = stream->ext_oneapi_submit_barrier()));
    CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));
    float allreduce_ms = 0;
    allreduce_ms =
        (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
         start->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;

  stream->parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
       dummy_kernel();
      });
    // warm up
    for (int i = 0; i < warmup_iters; i++) {
      fa.allreduce<T>(stream, self_data, result, data_size, threads,
                      block_limit);
    }
    /*
    DPCT1024:56: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CUDACHECK(DPCT_CHECK_ERROR(*start = stream->ext_oneapi_submit_barrier()));
    for (int i = 0; i < num_iters; i++) {
      fa.allreduce<T>(stream, self_data, result, data_size, threads,
                      block_limit);
    }
    /*
    DPCT1024:57: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    CUDACHECK(DPCT_CHECK_ERROR(*stop = stream->ext_oneapi_submit_barrier()));
    CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));

    float duration_ms = 0;
    duration_ms =
        (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
         start->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;
    if (myRank == 0)
      printf(
          "Rank %d done, nGPUs:%d, sz (kb): %d, %d, %d, my time:%.2fus, nccl "
          "time:%.2fus\n",
          myRank, nRanks, data_size * sizeof(T) / 1024, threads, block_limit,
          duration_ms * 1e3 / num_iters, allreduce_ms * 1e3 / num_iters);

    // And wait for all the queued up work to complete
    CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));

    NCCLCHECK(ncclAllReduce(self_data_copy, self_data, data_size, ncclDtype,
                            ncclSum, comm, stream));

    /*
    DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
   dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

   stream->parallel_for(
       sycl::nd_range<3>(sycl::range<3>(1, 1, 108) * sycl::range<3>(1, 1, 1024),
                         sycl::range<3>(1, 1, 1024)),
       [=](sycl::nd_item<3> item_ct1) {
        convert_data<T>(self_data, result, nccl_result, my_result, data_size,
                        item_ct1);
       });
  }
    CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));

    for (unsigned long j = 0; j < data_size; j++) {
      auto diff = abs(nccl_result[j] - my_result[j]);
      if (diff >= 4e-2) {
        printf("Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
               myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
        break;
      }
    }
    long double nccl_diffs = 0.0;
    long double my_diffs = 0.0;
    for (int j = 0; j < data_size; j++) {
      nccl_diffs += abs(nccl_result[j] - ground_truth[j]);
      my_diffs += abs(my_result[j] - ground_truth[j]);
    }
    if (myRank == 0)
      std::cout << "average abs diffs: nccl: " << nccl_diffs / data_size
                << " me: " << my_diffs / data_size << std::endl;
  } else {
    for (int i = 0; i < 100; i++) {
      fa.allreduce<T>(stream, self_data, result, data_size, threads,
                      block_limit);
      CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));
      NCCLCHECK(ncclAllReduce(self_data, self_data_copy, data_size, ncclDtype,
                              ncclSum, comm, stream));
      /*
      DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
   {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 108) *
                                               sycl::range<3>(1, 1, 1024),
                                           sycl::range<3>(1, 1, 1024)),
                         [=](sycl::nd_item<3> item_ct1) {
                          convert_data<T>(self_data_copy, result, nccl_result,
                                          my_result, data_size, item_ct1);
                         });
   }
      CUDACHECK(DPCT_CHECK_ERROR(stream->wait()));

      for (unsigned long j = 0; j < data_size; j++) {
        auto diff = abs(nccl_result[j] - my_result[j]);
        if (diff >= 4e-2) {
          printf(
              "Rank %d: Verification mismatch at %lld: %f != (my) %f, gt=%f\n",
              myRank, j, nccl_result[j], my_result[j], ground_truth[j]);
          break;
        }
      }
    }
    if (myRank == 0)
      printf("Test passed: nGPUs:%d, sz (kb): %d, %d, %d\n", nRanks,
             data_size * sizeof(T) / 1024, threads, block_limit);
    // long double nccl_diffs = 0.0;
    // long double my_diffs = 0.0;
    // for (int j = 0; j < data_size; j++) {
    //   nccl_diffs += abs(nccl_result[j] - ground_truth[j]);
    //   my_diffs += abs(my_result[j] - ground_truth[j]);
    // }
    // if (myRank == 0)
    //   std::cout << "average abs diffs: nccl: " << nccl_diffs / data_size
    //             << " me: " << my_diffs / data_size << std::endl;
  }

  CUDACHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(result, dpct::get_in_order_queue())));
  CUDACHECK(DPCT_CHECK_ERROR(
      dpct::dpct_free(self_data_copy, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(rank_data, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(buffer, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(states, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(sycl::free(ground_truth, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(sycl::free(nccl_result, dpct::get_in_order_queue())));
  CUDACHECK(
      DPCT_CHECK_ERROR(sycl::free(my_result, dpct::get_in_order_queue())));
  CUDACHECK(DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)));
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char** argv) try {
  int nRanks, myRank;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  /*
  DPCT1093:60: The "myRank" device may be not the one intended for use. Adjust
  the selected device if needed.
  */
  CUDACHECK(DPCT_CHECK_ERROR(dpct::select_device(myRank)));
  ncclUniqueId id;
  ncclComm_t comm;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0,
                     MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  bool performance_test = true;
  /*
  DPCT1026:58: The call to cudaProfilerStart was removed because SYCL currently
  does not support this function. Remove the profiler API will not impact the
  outcome.
  */
// Uncomment to scan through different block size configs.
// for (int threads : {256, 512, 1024}) {
//   for (int block_limit = 16; block_limit < 112; block_limit += 4) {
//     run<half>(myRank, nRanks, comm, threads, block_limit, 1024 * 1024,
//     performance_test);
//   }
// }
#ifdef USE_ROCM
  const int block_limit = 16;
#else
  const int block_limit = 36;
#endif
  // Scan through different sizes to test performance.
  for (int sz = 512; sz <= (8 << 20); sz *= 2) {
    run<sycl::half>(myRank, nRanks, comm, 512, 36, sz + 8 * 47,
                    performance_test);
  }

  /*
  DPCT1026:59: The call to cudaProfilerStop was removed because SYCL currently
  does not support this function. Remove the profiler API will not impact the
  outcome.
  */
  MPICHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}