#pragma once

// Minimal execution-space attribute macros for consumers compiled with
// `--no-offload-wrapper-inc` (zero CUDA SDK headers included textually).
#ifndef __device__
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif
