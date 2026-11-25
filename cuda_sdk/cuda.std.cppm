module;
#include "cuda_attributes.h"
#include <__clang_cuda_libdevice_declares.h>

// Supply the handful of CUDA bit-intrinsics that `<cuda/std/bit>` calls when
// `--no-offload-wrapper-inc` suppresses `__clang_cuda_device_functions.h`.
__device__ __forceinline__ int __ffs(int x) { return ::__nv_ffs(x); }
__device__ __forceinline__ int __ffsll(long long x) { return ::__nv_ffsll(x); }
__device__ __forceinline__ int __clz(int x) { return ::__nv_clz(x); }
__device__ __forceinline__ int __clzll(long long x) { return ::__nv_clzll(x); }
__device__ __forceinline__ int __popc(unsigned int x) {
  return ::__nv_popc(static_cast<int>(x));
}
__device__ __forceinline__ int __popcll(unsigned long long x) {
  return ::__nv_popcll(static_cast<long long>(x));
}
__device__ __forceinline__ unsigned int __brev(unsigned int x) {
  return static_cast<unsigned int>(::__nv_brev(static_cast<int>(x)));
}
__device__ __forceinline__ unsigned long long __brevll(unsigned long long x) {
  return static_cast<unsigned long long>(
      ::__nv_brevll(static_cast<long long>(x)));
}

#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

extern "C" {
__device__ int vprintf(const char *, const char *);
__device__ void *malloc(decltype(sizeof(0)));
__device__ void free(void *);
}

export module cuda.std;

#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
#define CUDA_STD_FAST_OR_SLOW(fast, slow) fast
#else
#define CUDA_STD_FAST_OR_SLOW(fast, slow) slow
#endif

#if defined(_CCCL_BEGIN_NAMESPACE_CUDA_STD)
#define CUDA_STD_NS_BEGIN _CCCL_BEGIN_NAMESPACE_CUDA_STD
#define CUDA_STD_NS_END _CCCL_END_NAMESPACE_CUDA_STD
#else
#define CUDA_STD_NS_BEGIN _LIBCUDACXX_BEGIN_NAMESPACE_STD
#define CUDA_STD_NS_END _LIBCUDACXX_END_NAMESPACE_STD
#endif

// Re-export core libcu++ (`cuda::std::*`) vocabulary types/functors alongside
// external-linkage `inline __device__` math, bit, sync, and memory primitives
// (avoiding legacy `static __device__` `-WTU-local-entity-exposure` warnings).
export CUDA_STD_NS_BEGIN

using ::cuda::std::array;
using ::cuda::std::span;
using ::cuda::std::optional;
using ::cuda::std::nullopt;
using ::cuda::std::pair;
using ::cuda::std::make_pair;
using ::cuda::std::tuple;
using ::cuda::std::make_tuple;
using ::cuda::std::get;
using ::cuda::std::plus;
using ::cuda::std::minus;
using ::cuda::std::multiplies;
using ::cuda::std::numeric_limits;
using ::cuda::std::move;
using ::cuda::std::forward;
using ::cuda::std::swap;

// --- Device math (libdevice, external C++ inline linkage) ---
__device__ __forceinline__ int abs(int x) { return ::__nv_abs(x); }
__device__ __forceinline__ long long abs(long long x) { return ::__nv_llabs(x); }
__device__ __forceinline__ float abs(float x) { return ::__nv_fabsf(x); }
__device__ __forceinline__ double abs(double x) { return ::__nv_fabs(x); }

__device__ __forceinline__ float fabs(float x) { return ::__nv_fabsf(x); }
__device__ __forceinline__ double fabs(double x) { return ::__nv_fabs(x); }

__device__ __forceinline__ float sqrt(float x) { return ::__nv_sqrtf(x); }
__device__ __forceinline__ double sqrt(double x) { return ::__nv_sqrt(x); }
__device__ __forceinline__ float rsqrt(float x) { return ::__nv_rsqrtf(x); }
__device__ __forceinline__ double rsqrt(double x) { return ::__nv_rsqrt(x); }

__device__ __forceinline__ float exp(float x) {
  return CUDA_STD_FAST_OR_SLOW(::__nv_fast_expf(x), ::__nv_expf(x));
}
__device__ __forceinline__ double exp(double x) { return ::__nv_exp(x); }

__device__ __forceinline__ float log(float x) {
  return CUDA_STD_FAST_OR_SLOW(::__nv_fast_logf(x), ::__nv_logf(x));
}
__device__ __forceinline__ double log(double x) { return ::__nv_log(x); }

__device__ __forceinline__ float sin(float x) {
  return CUDA_STD_FAST_OR_SLOW(::__nv_fast_sinf(x), ::__nv_sinf(x));
}
__device__ __forceinline__ double sin(double x) { return ::__nv_sin(x); }

__device__ __forceinline__ float cos(float x) {
  return CUDA_STD_FAST_OR_SLOW(::__nv_fast_cosf(x), ::__nv_cosf(x));
}
__device__ __forceinline__ double cos(double x) { return ::__nv_cos(x); }

__device__ __forceinline__ float tanh(float x) { return ::__nv_tanhf(x); }
__device__ __forceinline__ double tanh(double x) { return ::__nv_tanh(x); }

__device__ __forceinline__ float pow(float x, float y) {
  return CUDA_STD_FAST_OR_SLOW(::__nv_fast_powf(x, y), ::__nv_powf(x, y));
}
__device__ __forceinline__ double pow(double x, double y) {
  return ::__nv_pow(x, y);
}

__device__ __forceinline__ float fma(float a, float b, float c) {
  return ::__nv_fmaf(a, b, c);
}
__device__ __forceinline__ double fma(double a, double b, double c) {
  return ::__nv_fma(a, b, c);
}

__device__ __forceinline__ float fmin(float a, float b) {
  return ::__nv_fminf(a, b);
}
__device__ __forceinline__ double fmin(double a, double b) {
  return ::__nv_fmin(a, b);
}
__device__ __forceinline__ float fmax(float a, float b) {
  return ::__nv_fmaxf(a, b);
}
__device__ __forceinline__ double fmax(double a, double b) {
  return ::__nv_fmax(a, b);
}

__device__ __forceinline__ float floor(float x) { return ::__nv_floorf(x); }
__device__ __forceinline__ double floor(double x) { return ::__nv_floor(x); }
__device__ __forceinline__ float ceil(float x) { return ::__nv_ceilf(x); }
__device__ __forceinline__ double ceil(double x) { return ::__nv_ceil(x); }

__device__ __forceinline__ bool isnan(float x) { return ::__nv_isnanf(x) != 0; }
__device__ __forceinline__ bool isnan(double x) { return ::__nv_isnand(x) != 0; }
__device__ __forceinline__ bool isinf(float x) { return ::__nv_isinff(x); }
__device__ __forceinline__ bool isinf(double x) { return ::__nv_isinfd(x); }
__device__ __forceinline__ bool isfinite(float x) {
  return ::__nv_finitef(x) != 0;
}
__device__ __forceinline__ bool isfinite(double x) {
  return ::__nv_isfinited(x) != 0;
}

// --- Bit manipulation, sync, warp & device memory ---
__device__ __forceinline__ int popc(unsigned int x) {
  return ::__nv_popc(static_cast<int>(x));
}
__device__ __forceinline__ int clz(int x) { return ::__nv_clz(x); }
__device__ __forceinline__ int ffs(int x) { return ::__nv_ffs(x); }
__device__ __forceinline__ void syncthreads() { __syncthreads(); }
__device__ __forceinline__ void syncwarp(unsigned int mask = 0xffffffffu) {
  ::__nvvm_bar_warp_sync(mask);
}
__device__ __forceinline__ void *device_malloc(decltype(sizeof(0)) bytes) {
  return ::malloc(bytes);
}
__device__ __forceinline__ void device_free(void *ptr) { ::free(ptr); }

CUDA_STD_NS_END
