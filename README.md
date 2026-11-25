# CUDA C++20 Modules & PCH Examples

End-to-end CMake + Ninja examples demonstrating Clang's CUDA precompiled headers
and C++20 named modules (`--precompile`, `-fmodule-file`, `-include-pch`,
`-x cuda-module`, `--no-offload-wrapper-inc`, and per-side
`--offload-{host,device}-only` / `-Xarch_<arch>` workflows).

## Layout

* `cmake/CudaCXXModules.cmake` -- Reusable CMake helpers (`add_cuda_cxx_module`
  and `add_cuda_cxx_executable`) wiring `1 + N` host/per-arch BMIs and static
  module archives cleanly in Ninja
* `vectoradd/` -- Pure-module CUDA SAXPY example (zero `#include`s):
  * `math_helper.ixx` -- `.ixx` module interface compiled via `-x cuda-module`
  * `vecadd.cppm` -- `import std; import math_helper;` (exports `__global__`
    kernel whose SASS fatbin is cached once in `libvecadd_mod.a`)
  * `main.cu` -- `import std; import vecadd;`
* `cutlass_gemm/` -- Non-invasive CUTLASS + CuTe wrapper modules (zero changes
  to upstream CUTLASS, demonstrating both `.pcm` AST caching and `module.o`
  SASS fatbin caching):
  * `cutlass.cute.cppm` -- `export module cutlass.cute;`
  * `cutlass.gemm.cppm` -- `export module cutlass.gemm;`
  * `cutlass.sgemm_op.cppm` -- `export module cutlass.sgemm_op;` (instantiates
    the concrete SGEMM + CuTe kernels once into `cutlass.sgemm_op.o` so
    consumers skip both frontend parsing and `ptxas` SASS codegen)
  * `basic_gemm.cu` -- consumer instantiating generic `cutlass::gemm::device::Gemm`
  * `cached_gemm.cu` -- consumer importing `cutlass.sgemm_op` (zero device
    compilation in the consumer `.o`)
* `cuda_sdk/` -- Modularized CUDA Toolkit / CCCL ecosystem (`import cuda.*`):
  * `cuda.std.cppm` & `cuda.std.compat.cppm` -- mirrors `std` / `std.compat` on
    top of `libcu++` + `extern "C"` `libdevice`
  * `cuda.runtime.cppm`, `cuda.cub.cppm`, `cuda.thrust.cppm`, `cuda.cublas.cppm`
  * `main.cu` -- compiled with `--no-offload-wrapper-inc` (zero textual CUDA
    SDK prelude)

## Building

Set standard environment variables (`CXX`, `CXXFLAGS`, `LDFLAGS`, `PATH`,
`CUTLASS_ROOT`) as needed for your environment, then build with CMake + Ninja:

```bash
cmake -B build -G Ninja
cmake --build build
```

## Compile-Time Benchmark (`bench.py`)

Consumer `-O3 -c` wall-clock compile times on the CUTLASS + CuTe SGEMM example
(24-core x86_64 Linux host, prebuilt BMIs, unmodified upstream CUTLASS and
libstdc++ 15 `std` module):

| Configuration (`-O3 -c` consumer wall time) | `sm_80` | `sm_80` + `sm_90` |
|---|---:|---:|
| `nvcc`, textual `#include` | 12.80s | 17.20s |
| `clang++`, textual `#include` | 8.28s | 12.71s |
| `clang++`, `import std` + `import cutlass.{cute,gemm}` *(AST cached in `.pcm`)* | 2.35s (**3.5x**) | 3.75s (**3.4x**) |
| `clang++`, `--no-offload-wrapper-inc` + `import std` + `import cuda.runtime` + `import cutlass.sgemm_op` *(AST cached in `.pcm` + SASS fatbin cached in `module.o`)* | **0.88s (9.4x)** | **1.01s (12.6x)** |

* **`basic_gemm.cu` (`cutlass.{cute,gemm}`):** caches parsed CUTLASS/CuTe/STL
  ASTs in per-arch `.pcm` BMIs while still instantiating custom template
  specializations and running `ptxas` SASS codegen in the consumer TU.
* **`cached_gemm.cu` (`cutlass.sgemm_op`):** moves template instantiation +
  NVPTX + `ptxas` SASS compilation into a **one-time module object build**
  (`cutlass.sgemm_op.cppm -> cutlass.sgemm_op.o`: **2.13s** for `sm_80`,
  **3.43s** for `sm_80 + sm_90`), archived into `libcutlass_sgemm_op_mod.a`.
  The table above reports per-consumer `-c` compile time (excluding prebuilt
  `.pcm` and `module.o` artifacts):
  * For a **single cold consumer ($N = 1$)**, total compile time (`module.o` +
    `cached_gemm.o`) is comparable to `basic_gemm.cu` because `ptxas` still runs
    once inside `cutlass.sgemm_op.o`.
  * Across **$N$ consumers** sharing the operator (e.g. $3.43\text{s} + 1.01N$
    vs. $3.75N$ modular-AST or $12.71N$ textual on `sm_80 + sm_90`, breaking
    even at $N \ge 2$) and during **incremental rebuilds** of consumer TUs,
    each consumer TU skips `ptxas` completely (`cuobjdump` confirms
    `cached_gemm.cu.o` emits zero device code).

Reproduce locally with:

```bash
python3 bench.py
```
