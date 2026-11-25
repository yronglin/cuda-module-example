module;
import std.compat;
#ifndef CUBLASAPI
#define CUBLASAPI
#endif
#include <cublas_api.h>
export module cuda.cublas;

import cuda.runtime;

// Re-export core cuBLAS types/enums and expose the v2 entry points as real
// inline C++ functions (replacing cublas_v2.h's #define macros).
export {
  using ::cublasHandle_t;
  using ::cublasStatus_t;
  using ::cublasOperation_t;
  using ::CUBLAS_STATUS_SUCCESS;
  using ::CUBLAS_OP_N;
  using ::CUBLAS_OP_T;
  using ::CUBLAS_OP_C;

  inline cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    return ::cublasCreate_v2(handle);
  }
  inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    return ::cublasDestroy_v2(handle);
  }
  inline cublasStatus_t cublasSetStream(cublasHandle_t handle,
                                        cudaStream_t streamId) {
    return ::cublasSetStream_v2(handle, streamId);
  }
  inline cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                                    const float *alpha, const float *x,
                                    int incx, float *y, int incy) {
    return ::cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
  }
  inline cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float *x,
                                   int incx, const float *y, int incy,
                                   float *result) {
    return ::cublasSdot_v2(handle, n, x, incx, y, incy, result);
  }
  inline cublasStatus_t
  cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k, const float *alpha,
              const float *A, int lda, const float *B, int ldb,
              const float *beta, float *C, int ldc) {
    return ::cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  }
}

export namespace cuda::cublas {
using ::CUBLAS_OP_N;
using ::CUBLAS_OP_T;
using ::CUBLAS_STATUS_SUCCESS;
using Handle = ::cublasHandle_t;
using Status = ::cublasStatus_t;
using ::cublasCreate;
using ::cublasDestroy;
using ::cublasSaxpy;
using ::cublasSdot;
using ::cublasSetStream;
using ::cublasSgemm;
} // namespace cuda::cublas
