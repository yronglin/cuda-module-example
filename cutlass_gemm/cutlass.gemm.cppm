module;
#include "cutlass/gemm/device/gemm.h"
export module cutlass.gemm;

import cutlass.cute;

export namespace cutlass {

using ::cutlass::bfloat16_t;
using ::cutlass::half_t;
using ::cutlass::MatrixCoord;
using ::cutlass::Status;
using ::cutlass::TensorRef;

inline const char *status_string(Status s) {
  switch (s) {
  case Status::kSuccess:
    return "Success";
  case Status::kErrorMisalignedOperand:
    return "Error Misaligned Operand";
  case Status::kErrorInvalidDataType:
    return "Error Invalid Data Type";
  case Status::kErrorInvalidLayout:
    return "Error Invalid Layout";
  case Status::kErrorInvalidProblem:
    return "Error Invalid Problem";
  case Status::kErrorNotSupported:
    return "Error Not Supported";
  case Status::kErrorWorkspaceNull:
    return "Error Workspace Null";
  case Status::kErrorInternal:
    return "Error Internal";
  case Status::kErrorArchMismatch:
    return "Error Architecture Mismatch";
  case Status::kErrorInsufficientDriver:
    return "Error Insufficient Driver";
  case Status::kErrorMemoryAllocation:
    return "Error Memory Allocation failed";
  case Status::kInvalid:
    break;
  }
  return "Invalid status";
}

namespace layout {
using ::cutlass::layout::ColumnMajor;
using ::cutlass::layout::RowMajor;
} // namespace layout

namespace arch {
using ::cutlass::arch::OpClassSimt;
using ::cutlass::arch::OpClassTensorOp;
using ::cutlass::arch::Sm80;
} // namespace arch

namespace epilogue::thread {
using ::cutlass::epilogue::thread::LinearCombination;
} // namespace epilogue::thread

namespace gemm {
using ::cutlass::gemm::GemmCoord;
using ::cutlass::gemm::GemmShape;

namespace threadblock {
using ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
} // namespace threadblock

namespace device {
using ::cutlass::gemm::device::Gemm;
} // namespace device
} // namespace gemm

} // namespace cutlass
