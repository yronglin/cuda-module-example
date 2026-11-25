export module vecadd;

import std;
import math_helper;

export namespace vecadd {

__global__ void saxpy_kernel(float alpha, const float *x, const float *y,
                             float *out, std::size_t n) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = math_helper::saxpy_elem(alpha, x[i], y[i]);
}

cudaError_t launch_saxpy(float alpha, const float *x, const float *y,
                         float *out, std::size_t n) {
  constexpr unsigned int kBlock = 256;
  unsigned int grid = static_cast<unsigned int>((n + kBlock - 1) / kBlock);
  saxpy_kernel<<<grid, kBlock>>>(alpha, x, y, out, n);
  return cudaGetLastError();
}

} // namespace vecadd
