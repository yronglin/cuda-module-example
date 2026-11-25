module;
#include <cuda_runtime.h>
export module cuda.runtime;
import std;

export {
  using ::dim3;
  using ::uint3;
  using ::float4;
  using ::threadIdx;
  using ::blockIdx;
  using ::blockDim;
  using ::gridDim;

  using ::cudaError_t;
  using ::cudaError;
  using ::cudaSuccess;
  using ::cudaStream_t;
  using ::cudaEvent_t;
  using ::cudaMemcpyKind;
  using ::cudaMemcpyHostToHost;
  using ::cudaMemcpyHostToDevice;
  using ::cudaMemcpyDeviceToHost;
  using ::cudaMemcpyDeviceToDevice;
  using ::cudaMemcpyDefault;

  using ::cudaMemcpy;
  using ::cudaMemset;
  using ::cudaStreamCreate;
  using ::cudaStreamSynchronize;
  using ::cudaEventSynchronize;
  using ::cudaEventElapsedTime;
  using ::cudaDeviceSynchronize;
  using ::__cudaPushCallConfiguration;
}

// `<cuda_runtime.h>` and `<cuda_device_runtime_api.h>` add `static __inline__`
// overloads to names like `cudaMemcpyAsync` and `cudaMalloc`, so a blanket
// `export using ::cudaMemcpyAsync;` fails C++20's internal-linkage check.
// Redeclaring the `extern "C"` host C-ABI entry points directly exports only
// the external-linkage declarations cleanly!
export extern "C" {
  __host__ cudaError_t cudaMalloc(void **devPtr, std::size_t size);
  __host__ cudaError_t cudaFree(void *devPtr);
  __host__ cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                       std::size_t count,
                                       enum cudaMemcpyKind kind,
                                       cudaStream_t stream);
  __host__ cudaError_t cudaMemsetAsync(void *devPtr, int value,
                                       std::size_t count, cudaStream_t stream);
  __host__ cudaError_t cudaStreamDestroy(cudaStream_t stream);
  __host__ cudaError_t cudaEventCreate(cudaEvent_t *event);
  __host__ cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
  __host__ cudaError_t cudaEventDestroy(cudaEvent_t event);
  __host__ cudaError_t cudaGetLastError(void);
  __host__ const char *cudaGetErrorString(cudaError_t error);
  __host__ cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        std::size_t sharedMem,
                                        cudaStream_t stream);
}

export extern "C++" template <class T>
  requires(!__is_same(T, void))
inline __host__ cudaError_t cudaMalloc(T **devPtr, std::size_t size) {
  return ::cudaMalloc(reinterpret_cast<void **>(devPtr), size);
}
