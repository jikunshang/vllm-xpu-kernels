#include <torch/all.h>
#include <torch/cuda.h>
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

// This function assumes that `cpu_tensor` is a CPU tensor allocated with pinned
// memory, and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) try {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  // Get a device pointer corresponding to the pinned host memory
  void* device_ptr = nullptr;
  dpct::err0 err = DPCT_CHECK_ERROR(device_ptr = (void*)host_ptr);
  TORCH_CHECK(
      err == 0,
      /*
      DPCT1009:25: SYCL reports errors using exceptions and does not use error
      codes. Please replace the "get_error_string_dummy(...)" with a real
      error-handling function.
      */
      "cudaHostGetDevicePointer failed: ", dpct::get_error_string_dummy(err));

  // We'll use the same sizes, strides, and dtype as the CPU tensor.
  // TODO: check if layout is respected.
  auto sizes = cpu_tensor.sizes();
  auto strides = cpu_tensor.strides();
  auto options = cpu_tensor.options().device(torch::kCUDA);

  // from_blob signature: from_blob(void *data, IntArrayRef sizes, ..., Deleter,
  // const TensorOptions &) Provide a no-op deleter. The CPU tensor holds the
  // memory, so we don't free it here.
  auto deleter = [](void*) {
    // no-op, since the memory is owned by the original CPU tensor
  };

  torch::Tensor cuda_tensor =
      torch::from_blob(device_ptr, sizes, strides, deleter, options);

  TORCH_CHECK(cuda_tensor.device().is_cuda(),
              "Resulting tensor is not on CUDA device");

  return cuda_tensor;
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
