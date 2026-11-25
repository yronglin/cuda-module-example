export module math_helper;

export namespace math_helper {

inline __host__ __device__ float saxpy_elem(float alpha, float x, float y) {
  return alpha * x + y;
}

inline __device__ int compiled_sm_arch() {
#ifdef __CUDA_ARCH__
  return __CUDA_ARCH__;
#else
  return 0;
#endif
}

} // namespace math_helper
