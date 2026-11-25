module;

// `cute/config.hpp` gates `__host__ __device__` on `__CUDACC__`; enable it just
// across `cute/config.hpp` so Clang CUDA gets proper device attributes on CuTe.
#ifndef __CUDACC__
#define __CUDACC__ 1
#define CUTLASS_MOD_DEFINED_CUDACC 1
#endif
#include "cute/config.hpp"
#ifdef CUTLASS_MOD_DEFINED_CUDACC
#undef __CUDACC__
#undef CUTLASS_MOD_DEFINED_CUDACC
#endif

#include "cute/tensor.hpp"
export module cutlass.cute;

export namespace cute {

using ::cute::C;
using ::cute::Int;
using ::cute::_1;
using ::cute::_2;
using ::cute::_4;
using ::cute::_8;
using ::cute::_16;
using ::cute::_32;
using ::cute::_64;
using ::cute::_128;

using ::cute::Layout;
using ::cute::Shape;
using ::cute::Stride;
using ::cute::Tensor;

using ::cute::cosize;
using ::cute::local_partition;
using ::cute::local_tile;
using ::cute::make_coord;
using ::cute::make_gmem_ptr;
using ::cute::make_layout;
using ::cute::make_shape;
using ::cute::make_stride;
using ::cute::make_tensor;
using ::cute::size;

} // namespace cute
