#ifdef USE_TEXTUAL_CUTLASS
#include <cstdio>
#include <vector>
#ifndef __CUDACC__
#define __CUDACC__ 1
#define BASIC_GEMM_DEFINED_CUDACC 1
#endif
#include "cute/config.hpp"
#ifdef BASIC_GEMM_DEFINED_CUDACC
#undef __CUDACC__
#undef BASIC_GEMM_DEFINED_CUDACC
#endif
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm.h"
#else
import std;
import cutlass.cute;
import cutlass.gemm;
#endif

// Initialize a column-major matrix on device using CuTe `Tensor` + `Layout`
// from `import cutlass.cute;`.
__global__ void init_matrix_cute_kernel(float *ptr, int rows, int cols,
                                        int seed) {
  auto layout = cute::make_layout(cute::make_shape(rows, cols),
                                  cute::make_stride(cute::_1{}, rows));
  auto tensor = cute::make_tensor(cute::make_gmem_ptr(ptr), layout);

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < rows && j < cols) {
    int flat = i + j * rows;
    tensor(i, j) = static_cast<float>(((flat + seed) * 16807 % 16) - 8);
  }
}

// Launch single-precision GEMM using `cutlass::gemm::device::Gemm` from
// `import cutlass.gemm;`.
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta,
                           float *C, int ldc) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float,
                                  ColumnMajor>;

  CutlassGemm gemm_op;
  CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
                              {alpha, beta});
  if (gemm_op(args) != cutlass::Status::kSuccess)
    return cudaErrorUnknown;
  return cudaSuccess;
}

int main() {
  const int M = 256, N = 128, K = 64;
  float *A = nullptr, *B = nullptr, *C = nullptr;
  if (cudaMalloc(&A, sizeof(float) * M * K) != cudaSuccess ||
      cudaMalloc(&B, sizeof(float) * K * N) != cudaSuccess ||
      cudaMalloc(&C, sizeof(float) * M * N) != cudaSuccess) {
    std::printf("cudaMalloc failed (no GPU attached?)\n");
    return 1;
  }

  dim3 block(16, 16);
  init_matrix_cute_kernel<<<dim3((M + 15) / 16, (K + 15) / 16), block>>>(
      A, M, K, 1);
  init_matrix_cute_kernel<<<dim3((K + 15) / 16, (N + 15) / 16), block>>>(
      B, K, N, 2);
  cudaMemset(C, 0, sizeof(float) * M * N);

  cudaError_t err = CutlassSgemmNN(M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  if (err != cudaSuccess) {
    std::printf("CutlassSgemmNN failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::printf("cutlass.{cute,gemm} + import std basic_gemm OK (%dx%dx%d)\n", M,
              N, K);
  return 0;
}
