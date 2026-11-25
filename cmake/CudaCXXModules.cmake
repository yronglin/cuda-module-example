include_guard(GLOBAL)

function(_collect_cuda_module_deps OUT_FLAGS OUT_BMI_TARGETS)
  set(flags "")
  set(bmi_targets "")
  set(queue ${ARGN})
  set(visited "")
  while(queue)
    list(POP_FRONT queue cur)
    if(cur IN_LIST visited)
      continue()
    endif()
    list(APPEND visited "${cur}")

    get_target_property(mod_name "${cur}" CUDA_MODULE_NAME)
    get_target_property(mod_bmi "${cur}" CUDA_MODULE_HOST_BMI)
    get_target_property(mod_bmi_tgt "${cur}" CUDA_MODULE_BMI_TARGET)
    get_target_property(mod_deps "${cur}" CUDA_MODULE_IMPORTS)

    if(mod_name AND mod_bmi)
      list(APPEND flags "-fmodule-file=${mod_name}=${mod_bmi}")
    endif()
    if(mod_bmi_tgt)
      list(APPEND bmi_targets "${mod_bmi_tgt}")
    endif()
    if(mod_deps)
      list(APPEND queue ${mod_deps})
    endif()
  endwhile()
  set(${OUT_FLAGS} "${flags}" PARENT_SCOPE)
  set(${OUT_BMI_TARGETS} "${bmi_targets}" PARENT_SCOPE)
endfunction()

function(_get_active_cxx_flags OUT_VAR)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
  separate_arguments(_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
  set(${OUT_VAR} "${_flags}" PARENT_SCOPE)
endfunction()

# Query Clang's default C++ include search paths to locate `std` / `std.compat`.
function(find_cxx_std_module_sources OUT_STD_SRC OUT_STD_COMPAT_SRC)
  _get_active_cxx_flags(_cxx_flags)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" ${_cxx_flags} -x c++ -E -v -
    INPUT_FILE /dev/null
    OUTPUT_QUIET
    ERROR_VARIABLE _verbose_err
  )
  string(REGEX MATCHALL "\n ([^\n]+)" _lines "${_verbose_err}")
  set(std_src "")
  set(std_compat_src "")
  foreach(line IN LISTS _lines)
    string(STRIP "${line}" dir)
    if(EXISTS "${dir}/bits/std.cc")
      set(std_src "${dir}/bits/std.cc")
      set(std_compat_src "${dir}/bits/std.compat.cc")
      break()
    endif()
    if(EXISTS "${dir}/../../share/libc++/v1/std.cppm")
      get_filename_component(libcxx_dir "${dir}/../../share/libc++/v1" REALPATH)
      set(std_src "${libcxx_dir}/std.cppm")
      set(std_compat_src "${libcxx_dir}/std.compat.cppm")
      break()
    endif()
  endforeach()

  if(NOT std_src)
    message(FATAL_ERROR
      "Could not locate C++23 `std` module sources (`bits/std.cc` or `std.cppm`) "
      "from `${CMAKE_CXX_COMPILER}`'s default standard library.")
  endif()

  set(${OUT_STD_SRC} "${std_src}" PARENT_SCOPE)
  set(${OUT_STD_COMPAT_SRC} "${std_compat_src}" PARENT_SCOPE)
endfunction()

function(add_cuda_cxx_module TARGET_NAME)
  set(options NO_WRAPPER_INC BMI_ONLY)
  set(one_value_args MODULE_NAME SOURCE)
  set(multi_value_args INCLUDE_DIRS COMPILE_OPTIONS IMPORTS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  get_filename_component(src_abs "${ARG_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(bmi_dir "${CMAKE_CURRENT_BINARY_DIR}/bmi")
  file(MAKE_DIRECTORY "${bmi_dir}")

  set(host_bmi "${bmi_dir}/${ARG_MODULE_NAME}.pcm")
  set(all_bmis "${host_bmi}")
  foreach(arch IN LISTS CUDA_MODULE_ARCHS)
    list(APPEND all_bmis "${bmi_dir}/${ARG_MODULE_NAME}-cuda-nvptx64-nvidia-cuda-${arch}.pcm")
  endforeach()

  _collect_cuda_module_deps(dep_flags dep_bmi_targets ${ARG_IMPORTS})
  _get_active_cxx_flags(active_cxx_flags)

  set(inc_flags "")
  foreach(inc IN LISTS ARG_INCLUDE_DIRS)
    list(APPEND inc_flags "-I${inc}")
  endforeach()

  set(wrap_flag "")
  if(ARG_NO_WRAPPER_INC)
    set(wrap_flag "--no-offload-wrapper-inc")
  endif()

  add_custom_command(
    OUTPUT ${all_bmis}
    COMMAND "${CMAKE_CXX_COMPILER}"
            "-std=c++${CMAKE_CXX_STANDARD}"
            ${active_cxx_flags}
            ${ARG_COMPILE_OPTIONS}
            -x cuda-module
            "--cuda-path=${CUDAToolkit_TARGET_DIR}"
            ${wrap_flag}
            ${CUDA_ARCH_FLAGS}
            ${inc_flags}
            ${dep_flags}
            --precompile "${src_abs}"
            -o "${host_bmi}"
    DEPENDS "${src_abs}" ${dep_bmi_targets}
    COMMENT "Precompiling CUDA module ${ARG_MODULE_NAME}"
    VERBATIM
  )

  set(bmi_target "${TARGET_NAME}_bmi")
  add_custom_target("${bmi_target}" DEPENDS ${all_bmis})
  if(dep_bmi_targets)
    add_dependencies("${bmi_target}" ${dep_bmi_targets})
  endif()

  if(ARG_BMI_ONLY)
    add_library("${TARGET_NAME}" INTERFACE)
    add_dependencies("${TARGET_NAME}" "${bmi_target}")
  else()
    add_library("${TARGET_NAME}" STATIC "${src_abs}")
    set_source_files_properties("${src_abs}" PROPERTIES LANGUAGE CXX)
    add_dependencies("${TARGET_NAME}" "${bmi_target}")
    if(dep_bmi_targets)
      add_dependencies("${TARGET_NAME}" ${dep_bmi_targets})
    endif()
    target_compile_options("${TARGET_NAME}" PRIVATE
      ${ARG_COMPILE_OPTIONS}
      -x cuda-module
      "--cuda-path=${CUDAToolkit_TARGET_DIR}"
      ${wrap_flag}
      ${CUDA_ARCH_FLAGS}
      ${CUDA_PTX_FLAGS}
      ${inc_flags}
      ${dep_flags}
    )
    if(ARG_IMPORTS)
      target_link_libraries("${TARGET_NAME}" PUBLIC ${ARG_IMPORTS})
    endif()
  endif()

  set_target_properties("${TARGET_NAME}" PROPERTIES
    CUDA_MODULE_NAME "${ARG_MODULE_NAME}"
    CUDA_MODULE_HOST_BMI "${host_bmi}"
    CUDA_MODULE_BMI_TARGET "${bmi_target}"
    CUDA_MODULE_IMPORTS "${ARG_IMPORTS}"
  )
endfunction()

function(add_cuda_cxx_executable TARGET_NAME)
  set(options NO_WRAPPER_INC)
  set(one_value_args "")
  set(multi_value_args SOURCES INCLUDE_DIRS COMPILE_OPTIONS IMPORTS LINK_LIBS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_executable("${TARGET_NAME}" ${ARG_SOURCES})
  set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE CXX)

  _collect_cuda_module_deps(dep_flags dep_bmi_targets ${ARG_IMPORTS})
  if(dep_bmi_targets)
    add_dependencies("${TARGET_NAME}" ${dep_bmi_targets})
  endif()

  set(inc_flags "")
  foreach(inc IN LISTS ARG_INCLUDE_DIRS)
    list(APPEND inc_flags "-I${inc}")
  endforeach()

  set(wrap_flag "")
  if(ARG_NO_WRAPPER_INC)
    set(wrap_flag "--no-offload-wrapper-inc")
  endif()

  target_compile_options("${TARGET_NAME}" PRIVATE
    ${ARG_COMPILE_OPTIONS}
    -x cuda
    "--cuda-path=${CUDAToolkit_TARGET_DIR}"
    ${wrap_flag}
    ${CUDA_ARCH_FLAGS}
    ${CUDA_PTX_FLAGS}
    ${inc_flags}
    ${dep_flags}
  )
  target_link_libraries("${TARGET_NAME}" PRIVATE
    ${ARG_IMPORTS}
    ${ARG_LINK_LIBS}
    CUDA::cudart_static
  )
endfunction()
