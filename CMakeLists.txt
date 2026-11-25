cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(cuda_module_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(CUDAToolkit REQUIRED)

set(CUDA_MODULE_ARCHS "sm_80;sm_90" CACHE STRING "Target CUDA GPU architectures")
list(GET CUDA_MODULE_ARCHS -1 _last_arch)
set(CUDA_PTX_FLAGS "--cuda-include-ptx=${_last_arch}")

set(CUDA_ARCH_FLAGS "")
foreach(arch IN LISTS CUDA_MODULE_ARCHS)
  list(APPEND CUDA_ARCH_FLAGS "--cuda-gpu-arch=${arch}")
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CudaCXXModules.cmake")
find_cxx_std_module_sources(STD_MODULE_SOURCE STD_COMPAT_MODULE_SOURCE)

# 0. C++23 `std` and `std.compat`
add_cuda_cxx_module(std_mod
  BMI_ONLY
  NO_WRAPPER_INC
  MODULE_NAME "std"
  SOURCE "${STD_MODULE_SOURCE}"
  COMPILE_OPTIONS -Wno-reserved-module-identifier
)

add_cuda_cxx_module(std_compat_mod
  BMI_ONLY
  NO_WRAPPER_INC
  MODULE_NAME "std.compat"
  SOURCE "${STD_COMPAT_MODULE_SOURCE}"
  COMPILE_OPTIONS -Wno-reserved-module-identifier
  IMPORTS std_mod
)

# 1. vectoradd (`.ixx` + `.cppm` + `main.cu`)
add_cuda_cxx_module(math_helper_mod
  MODULE_NAME "math_helper"
  SOURCE "vectoradd/math_helper.ixx"
)

add_cuda_cxx_module(vecadd_mod
  MODULE_NAME "vecadd"
  SOURCE "vectoradd/vecadd.cppm"
  IMPORTS std_mod math_helper_mod
)

add_cuda_cxx_executable(vectoradd
  SOURCES "vectoradd/main.cu"
  IMPORTS std_mod math_helper_mod vecadd_mod
)

# 2. cuda_sdk_demo (`--no-offload-wrapper-inc` + `cuda.{std,std.compat,runtime,cub,thrust,cublas}`)
set(CUDA_SDK_INC "${CMAKE_CURRENT_SOURCE_DIR}/cuda_sdk")

add_cuda_cxx_module(cuda_std_mod
  NO_WRAPPER_INC
  MODULE_NAME "cuda.std"
  SOURCE "cuda_sdk/cuda.std.cppm"
  INCLUDE_DIRS "${CUDA_SDK_INC}"
  COMPILE_OPTIONS -Werror=TU-local-entity-exposure
)

add_cuda_cxx_module(cuda_std_compat_mod
  NO_WRAPPER_INC
  MODULE_NAME "cuda.std.compat"
  SOURCE "cuda_sdk/cuda.std.compat.cppm"
  INCLUDE_DIRS "${CUDA_SDK_INC}"
  COMPILE_OPTIONS -Werror=TU-local-entity-exposure
  IMPORTS std_mod std_compat_mod cuda_std_mod
)

add_cuda_cxx_module(cuda_runtime_mod
  MODULE_NAME "cuda.runtime"
  SOURCE "cuda_sdk/cuda.runtime.cppm"
  IMPORTS std_mod
)

add_cuda_cxx_module(cuda_cub_mod
  MODULE_NAME "cuda.cub"
  SOURCE "cuda_sdk/cuda.cub.cppm"
  IMPORTS cuda_std_mod cuda_runtime_mod
)

add_cuda_cxx_module(cuda_thrust_mod
  MODULE_NAME "cuda.thrust"
  SOURCE "cuda_sdk/cuda.thrust.cppm"
  IMPORTS cuda_std_mod cuda_runtime_mod cuda_cub_mod
)

add_cuda_cxx_module(cuda_cublas_mod
  MODULE_NAME "cuda.cublas"
  SOURCE "cuda_sdk/cuda.cublas.cppm"
  IMPORTS std_mod std_compat_mod cuda_runtime_mod
)

add_cuda_cxx_executable(cuda_sdk_demo
  NO_WRAPPER_INC
  SOURCES "cuda_sdk/main.cu"
  INCLUDE_DIRS "${CUDA_SDK_INC}"
  COMPILE_OPTIONS -Werror=TU-local-entity-exposure
  IMPORTS std_mod std_compat_mod cuda_std_mod cuda_std_compat_mod
          cuda_runtime_mod cuda_cub_mod cuda_thrust_mod cuda_cublas_mod
  LINK_LIBS CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos
)

# 3. CUTLASS + CuTe (built automatically when CUTLASS headers are discoverable)
find_path(CUTLASS_INCLUDE_DIR
  NAMES cutlass/cutlass.h
  HINTS ENV CUTLASS_ROOT
  PATH_SUFFIXES include
)

if(CUTLASS_INCLUDE_DIR)
  add_cuda_cxx_module(cutlass_cute_mod
    MODULE_NAME "cutlass.cute"
    SOURCE "cutlass_gemm/cutlass.cute.cppm"
    INCLUDE_DIRS "${CUTLASS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm"
  )

  add_cuda_cxx_module(cutlass_gemm_mod
    MODULE_NAME "cutlass.gemm"
    SOURCE "cutlass_gemm/cutlass.gemm.cppm"
    INCLUDE_DIRS "${CUTLASS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm"
    IMPORTS cutlass_cute_mod
  )

  # Instantiates the concrete SGEMM + CuTe kernels once into `libcutlass_sgemm_op_mod.a`
  # so consumers skip both frontend parsing and backend `ptxas` SASS codegen.
  add_cuda_cxx_module(cutlass_sgemm_op_mod
    MODULE_NAME "cutlass.sgemm_op"
    SOURCE "cutlass_gemm/cutlass.sgemm_op.cppm"
    INCLUDE_DIRS "${CUTLASS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm"
    IMPORTS cutlass_cute_mod cutlass_gemm_mod
  )

  add_cuda_cxx_executable(cutlass_gemm
    SOURCES "cutlass_gemm/basic_gemm.cu"
    INCLUDE_DIRS "${CUTLASS_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cutlass_gemm"
    IMPORTS std_mod cutlass_cute_mod cutlass_gemm_mod
  )

  add_cuda_cxx_executable(cutlass_cached_gemm
    NO_WRAPPER_INC
    SOURCES "cutlass_gemm/cached_gemm.cu"
    IMPORTS std_mod cuda_runtime_mod cutlass_sgemm_op_mod
  )
endif()

# 4. Build-system per-side (`--offload-{host,device}-only`) + `-Xarch_<arch>` demo
list(LENGTH CUDA_MODULE_ARCHS _num_archs)
if(_num_archs GREATER_EQUAL 2)
  list(GET CUDA_MODULE_ARCHS 0 _arch0)
  list(GET CUDA_MODULE_ARCHS 1 _arch1)
  set(PER_SIDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/per_side")
  get_target_property(STD_HOST_BMI std_mod CUDA_MODULE_HOST_BMI)
  get_filename_component(STD_BMI_DIR "${STD_HOST_BMI}" DIRECTORY)
  _get_active_cxx_flags(_per_side_cxx_flags)

  set(_cuda_path_flag "--cuda-path=${CUDAToolkit_TARGET_DIR}")
  add_custom_command(
    OUTPUT "${PER_SIDE_DIR}/main.o"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${PER_SIDE_DIR}/host" "${PER_SIDE_DIR}/${_arch0}" "${PER_SIDE_DIR}/${_arch1}"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda-module --offload-host-only
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/math_helper.ixx"
            -o "${PER_SIDE_DIR}/host/math_helper.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda-module --offload-device-only "--cuda-gpu-arch=${_arch0}"
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/math_helper.ixx"
            -o "${PER_SIDE_DIR}/${_arch0}/math_helper.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda-module --offload-device-only "--cuda-gpu-arch=${_arch1}"
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/math_helper.ixx"
            -o "${PER_SIDE_DIR}/${_arch1}/math_helper.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda --offload-host-only
            "-fmodule-file=std=${STD_BMI_DIR}/std.pcm"
            "-fmodule-file=math_helper=${PER_SIDE_DIR}/host/math_helper.pcm"
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/vecadd.cppm"
            -o "${PER_SIDE_DIR}/host/vecadd.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda --offload-device-only "--cuda-gpu-arch=${_arch0}"
            "-fmodule-file=std=${STD_BMI_DIR}/std-cuda-nvptx64-nvidia-cuda-${_arch0}.pcm"
            "-fmodule-file=math_helper=${PER_SIDE_DIR}/${_arch0}/math_helper.pcm"
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/vecadd.cppm"
            -o "${PER_SIDE_DIR}/${_arch0}/vecadd.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda --offload-device-only "--cuda-gpu-arch=${_arch1}"
            "-fmodule-file=std=${STD_BMI_DIR}/std-cuda-nvptx64-nvidia-cuda-${_arch1}.pcm"
            "-fmodule-file=math_helper=${PER_SIDE_DIR}/${_arch1}/math_helper.pcm"
            --precompile "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/vecadd.cppm"
            -o "${PER_SIDE_DIR}/${_arch1}/vecadd.pcm"
    COMMAND "${CMAKE_CXX_COMPILER}" -std=c++20 ${_per_side_cxx_flags} ${_cuda_path_flag} -x cuda
            "--cuda-gpu-arch=${_arch0}" "--cuda-gpu-arch=${_arch1}"
            "-fmodule-file=std=${STD_HOST_BMI}"
            -Xarch_host        "-fmodule-file=math_helper=${PER_SIDE_DIR}/host/math_helper.pcm"
            "-Xarch_${_arch0}" "-fmodule-file=math_helper=${PER_SIDE_DIR}/${_arch0}/math_helper.pcm"
            "-Xarch_${_arch1}" "-fmodule-file=math_helper=${PER_SIDE_DIR}/${_arch1}/math_helper.pcm"
            -Xarch_host        "-fmodule-file=vecadd=${PER_SIDE_DIR}/host/vecadd.pcm"
            "-Xarch_${_arch0}" "-fmodule-file=vecadd=${PER_SIDE_DIR}/${_arch0}/vecadd.pcm"
            "-Xarch_${_arch1}" "-fmodule-file=vecadd=${PER_SIDE_DIR}/${_arch1}/vecadd.pcm"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/main.cu"
            -o "${PER_SIDE_DIR}/main.o"
    DEPENDS std_mod_bmi
            "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/math_helper.ixx"
            "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/vecadd.cppm"
            "${CMAKE_CURRENT_SOURCE_DIR}/vectoradd/main.cu"
    COMMENT "Building per-side + -Xarch_<arch> demo object"
    VERBATIM
  )
  add_custom_target(per_side_demo ALL DEPENDS "${PER_SIDE_DIR}/main.o")
endif()
