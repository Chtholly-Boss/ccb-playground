#include "binding.cuh"

/// cpu addition
int add(int a, int b) { return a + b; }

/// cuda tensor addition

__global__ void add_kernel(float *out, const float *a, const float *b,
                           size_t size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = a[idx] + b[idx];
  }
}

void add_tensor(const nb::ndarray<> &out, const nb::ndarray<> &a,
                const nb::ndarray<> &b) {
  /// pre-condition checks in python side
  /// - dtype is float32
  /// - same shape
  /// - contiguous
  /// - device is cuda
  const size_t size = out.size();
  constexpr auto kThreadsPerBlock = 256;
  const size_t num_blocks = (size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  add_kernel<<<num_blocks, kThreadsPerBlock>>>(
      reinterpret_cast<float *>(out.data()),
      reinterpret_cast<const float *>(a.data()),
      reinterpret_cast<const float *>(b.data()), size);
}
