#ifdef USE_TEXTUAL_SDK
#include <cstdio>
#include <vector>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cublas_v2.h>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#else
#include "cuda_attributes.h"

import std;
import cuda.std;
import cuda.std.compat;
import cuda.runtime;
import cuda.cub;
import cuda.thrust;
import cuda.cublas;
#endif

// 1. Custom kernel using both `cuda.std` (`cuda::std::sqrt`) and
//    `cuda.std.compat` (global `fmaf`, `size_t`), `threadIdx`/`blockIdx` (from
//    `cuda.runtime`), and `cub::BlockReduce` (from `cuda.cub`).
template <int BlockThreads>
__global__ void block_sum_kernel(const float *in, float *block_sums, size_t n) {
  using BlockReduce = cub::BlockReduce<float, BlockThreads>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  size_t idx = static_cast<size_t>(blockIdx.x) * BlockThreads + threadIdx.x;
  float val = 0.0f;
  if (idx < n) {
#ifdef USE_TEXTUAL_SDK
    val = ::fmaf(in[idx], ::sqrtf(1.0f), 0.0f);
#else
    val = fmaf(in[idx], cuda::std::sqrt(1.0f), 0.0f);
#endif
  }
  float sum = BlockReduce(temp_storage).Sum(val);
  if (threadIdx.x == 0)
    block_sums[blockIdx.x] = sum;
}

int main() {
  constexpr int N = 1024;
  constexpr int kThreads = 256;
  constexpr int kBlocks = (N + kThreads - 1) / kThreads;

  // 2. `cuda.runtime`: stream creation
  cudaStream_t stream{};
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::printf("cudaStreamCreate failed (no GPU attached?)\n");
    return 1;
  }

  // 3. `cuda.thrust` + `cuda.std`: device_vector + `cuda::std::plus`
  thrust::device_vector<float> dx(N), dy(N, 1.0f), dblock_sums(kBlocks);
  thrust::sequence(dx.begin(), dx.end(), 1.0f);
  thrust::transform(dx.begin(), dx.end(), dy.begin(), dy.begin(),
                    cuda::std::plus<float>{});

  // 4. `cuda.cublas`: SAXPY + SDOT on raw device pointers
  cublasHandle_t blas{};
  cublasCreate(&blas);
  cublasSetStream(blas, stream);

  const float alpha = 2.0f;
  float dot_result = 0.0f;
  cublasSaxpy(blas, N, &alpha, thrust::raw_pointer_cast(dx.data()), 1,
              thrust::raw_pointer_cast(dy.data()), 1);
  cublasSdot(blas, N, thrust::raw_pointer_cast(dx.data()), 1,
             thrust::raw_pointer_cast(dy.data()), 1, &dot_result);

  // 5. `cuda.cub`: launch `cub::BlockReduce` kernel + `cub::DeviceReduce::Sum`
  block_sum_kernel<kThreads><<<kBlocks, kThreads, 0, stream>>>(
      thrust::raw_pointer_cast(dy.data()),
      thrust::raw_pointer_cast(dblock_sums.data()), N);

  float *d_total = nullptr;
  void *d_temp = nullptr;
  std::size_t temp_bytes = 0;
  cudaMalloc(&d_total, sizeof(float));
  cub::DeviceReduce::Sum(nullptr, temp_bytes,
                         thrust::raw_pointer_cast(dblock_sums.data()), d_total,
                         kBlocks, stream);
  cudaMalloc(&d_temp, temp_bytes);
  cub::DeviceReduce::Sum(d_temp, temp_bytes,
                         thrust::raw_pointer_cast(dblock_sums.data()), d_total,
                         kBlocks, stream);

  float host_total = 0.0f;
  cudaMemcpyAsync(&host_total, d_total, sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_temp);
  cudaFree(d_total);
  cublasDestroy(blas);
  cudaStreamDestroy(stream);

  float thrust_total = thrust::reduce(dy.begin(), dy.end(), 0.0f);

  std::printf(
      "std + cuda.{std,std.compat,runtime,cub,thrust,cublas} OK (cub=%.1f, "
      "thrust=%.1f, dot=%.1f)\n",
      host_total, thrust_total, dot_result);
  return 0;
}
