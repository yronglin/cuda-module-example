module;
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
export module cuda.thrust;

import cuda.std;
import cuda.runtime;
import cuda.cub;

export THRUST_NAMESPACE_BEGIN
  using THRUST_NS_QUALIFIER::device_ptr;
  using THRUST_NS_QUALIFIER::device_vector;
  using THRUST_NS_QUALIFIER::host_vector;
  using THRUST_NS_QUALIFIER::raw_pointer_cast;
  using THRUST_NS_QUALIFIER::fill;
  using THRUST_NS_QUALIFIER::sequence;
  using THRUST_NS_QUALIFIER::transform;
  using THRUST_NS_QUALIFIER::reduce;
  using THRUST_NS_QUALIFIER::plus;
  using THRUST_NS_QUALIFIER::multiplies;
THRUST_NAMESPACE_END

export namespace cuda::thrust {
inline constexpr THRUST_NS_QUALIFIER::detail::device_t device{};
inline constexpr THRUST_NS_QUALIFIER::detail::host_t host{};
using ::thrust::device_ptr;
using ::thrust::device_vector;
using ::thrust::fill;
using ::thrust::host_vector;
using ::thrust::multiplies;
using ::thrust::plus;
using ::thrust::raw_pointer_cast;
using ::thrust::reduce;
using ::thrust::sequence;
using ::thrust::transform;
} // namespace cuda::thrust
