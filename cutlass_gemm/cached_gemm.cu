// Consumer TU importing `std`, `cuda.runtime`, and the pre-instantiated
// operator module `cutlass.sgemm_op` under `--no-offload-wrapper-inc`.
// Zero textual headers (not even the CUDA prelude!) are parsed, and zero
// GPU SASS codegen runs in this TU.
import std;
import cuda.runtime;
import cutlass.sgemm_op;

int main() {
  const int M = 256, N = 128, K = 64;
  float *A = nullptr, *B = nullptr, *C = nullptr;
  if (cudaMalloc(&A, sizeof(float) * M * K) != cudaSuccess ||
      cudaMalloc(&B, sizeof(float) * K * N) != cudaSuccess ||
      cudaMalloc(&C, sizeof(float) * M * N) != cudaSuccess) {
    std::printf("cudaMalloc failed (no GPU attached?)\n");
    return 1;
  }

  cutlass::ops::init_matrix(A, M, K, 1);
  cutlass::ops::init_matrix(B, K, N, 2);
  cudaMemset(C, 0, sizeof(float) * M * N);

  cudaError_t err =
      cutlass::ops::sgemm_nn(M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  if (err != cudaSuccess) {
    std::printf("sgemm_nn failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::printf("cutlass.sgemm_op (cached AST + cached SASS .o) OK (%dx%dx%d)\n",
              M, N, K);
  return 0;
}
