export module cutlass.sgemm_op;

import cutlass.cute;
import cutlass.gemm;

// Concrete GEMM operator instantiated and compiled ONCE into `cutlass.sgemm_op.o`
// (including its sm_80 + sm_90 SASS fatbin!). Consumers importing
// `cutlass.sgemm_op` skip BOTH frontend CUTLASS header parsing AND backend
// `ptxas` SASS compilation of the GEMM kernel.
export namespace cutlass::ops {

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

cudaError_t init_matrix(float *ptr, int rows, int cols, int seed) {
  dim3 block(16, 16);
  dim3 grid((rows + 15) / 16, (cols + 15) / 16);
  init_matrix_cute_kernel<<<grid, block>>>(ptr, rows, cols, seed);
  return cudaGetLastError();
}

cudaError_t sgemm_nn(int M, int N, int K, float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc) {
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

} // namespace cutlass::ops
