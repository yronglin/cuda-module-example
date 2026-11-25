module;
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
export module cuda.cub;

import cuda.std;
import cuda.runtime;

export CUB_NAMESPACE_BEGIN
  using CUB_NS_QUALIFIER::BlockLoad;
  using CUB_NS_QUALIFIER::BlockReduce;
  using CUB_NS_QUALIFIER::BlockScan;
  using CUB_NS_QUALIFIER::BlockStore;
  using CUB_NS_QUALIFIER::BlockReduceAlgorithm;
  using CUB_NS_QUALIFIER::BLOCK_REDUCE_WARP_REDUCTIONS;
  using CUB_NS_QUALIFIER::DeviceReduce;
  using CUB_NS_QUALIFIER::DeviceScan;
CUB_NAMESPACE_END

export namespace cuda::cub {
using ::cub::BlockLoad;
using ::cub::BlockReduce;
using ::cub::BlockScan;
using ::cub::BlockStore;
using ::cub::DeviceReduce;
using ::cub::DeviceScan;
} // namespace cuda::cub
