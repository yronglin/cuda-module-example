import std;
import vecadd;

int main() {
  constexpr std::size_t N = 1024;
  constexpr float alpha = 2.5f;

  std::vector<float> hx(N), hy(N), hout(N, 0.0f);
  for (std::size_t i = 0; i < N; ++i) {
    hx[i] = static_cast<float>(i);
    hy[i] = static_cast<float>(2 * i);
  }

  float *dx = nullptr, *dy = nullptr, *dout = nullptr;
  if (cudaMalloc(&dx, N * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dy, N * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dout, N * sizeof(float)) != cudaSuccess) {
    std::printf("cudaMalloc failed (no GPU attached?)\n");
    return 1;
  }

  cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  cudaError_t err = vecadd::launch_saxpy(alpha, dx, dy, dout, N);
  if (err != cudaSuccess) {
    std::printf("launch_saxpy failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(hout.data(), dout, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dout);

  for (std::size_t i = 0; i < N; ++i) {
    float expected = alpha * hx[i] + hy[i];
    if (std::abs(hout[i] - expected) > 1e-5f) {
      std::printf("Mismatch at %zu: got %f, expected %f\n", i, hout[i],
                  expected);
      return 1;
    }
  }
  std::printf("vectoradd OK (%zu elements)\n", N);
  return 0;
}
