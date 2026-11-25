module;
import std.compat;
#include "cuda_attributes.h"
#include <__clang_cuda_libdevice_declares.h>
export module cuda.std.compat;

export import cuda.std;

#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
#define CUDA_STD_COMPAT_FAST_OR_SLOW(fast, slow) fast
#else
#define CUDA_STD_COMPAT_FAST_OR_SLOW(fast, slow) slow
#endif

// Mirrors C++23 `std.compat`: re-exports `cuda.std` and exposes C / CUDA math
// and fixed-width integer names in the global namespace for device code.
export {
  using ::ptrdiff_t;
  using ::size_t;
  using ::int8_t;
  using ::int16_t;
  using ::int32_t;
  using ::int64_t;
  using ::uint8_t;
  using ::uint16_t;
  using ::uint32_t;
  using ::uint64_t;

  __device__ __forceinline__ float fabsf(float x) { return ::__nv_fabsf(x); }
  __device__ __forceinline__ float sqrtf(float x) { return ::__nv_sqrtf(x); }
  __device__ __forceinline__ float rsqrtf(float x) { return ::__nv_rsqrtf(x); }
  __device__ __forceinline__ float expf(float x) {
    return CUDA_STD_COMPAT_FAST_OR_SLOW(::__nv_fast_expf(x), ::__nv_expf(x));
  }
  __device__ __forceinline__ float logf(float x) {
    return CUDA_STD_COMPAT_FAST_OR_SLOW(::__nv_fast_logf(x), ::__nv_logf(x));
  }
  __device__ __forceinline__ float sinf(float x) {
    return CUDA_STD_COMPAT_FAST_OR_SLOW(::__nv_fast_sinf(x), ::__nv_sinf(x));
  }
  __device__ __forceinline__ float cosf(float x) {
    return CUDA_STD_COMPAT_FAST_OR_SLOW(::__nv_fast_cosf(x), ::__nv_cosf(x));
  }
  __device__ __forceinline__ float tanhf(float x) { return ::__nv_tanhf(x); }
  __device__ __forceinline__ float powf(float x, float y) {
    return CUDA_STD_COMPAT_FAST_OR_SLOW(::__nv_fast_powf(x, y),
                                        ::__nv_powf(x, y));
  }
  __device__ __forceinline__ float fmaf(float a, float b, float c) {
    return ::__nv_fmaf(a, b, c);
  }
  __device__ __forceinline__ float fminf(float a, float b) {
    return ::__nv_fminf(a, b);
  }
  __device__ __forceinline__ float fmaxf(float a, float b) {
    return ::__nv_fmaxf(a, b);
  }
  __device__ __forceinline__ float floorf(float x) { return ::__nv_floorf(x); }
  __device__ __forceinline__ float ceilf(float x) { return ::__nv_ceilf(x); }
}
