#!/usr/bin/env python3
"""Benchmark consumer compile times across textual #include, AST-cached BMIs,
and SASS-cached module object files (`cutlass.sgemm_op`)."""

import os
import re
import shlex
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def detect_std_module_source(clang, cxxflags):
    res = subprocess.run(
        [clang, *cxxflags, "-x", "c++", "-E", "-v", "-"],
        input=b"",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    for m in re.finditer(r"\n ([^\n]+)", res.stderr.decode("utf-8", "replace")):
        inc = m.group(1).strip()
        for candidate in [
            os.path.join(inc, "bits", "std.cc"),
            os.path.normpath(os.path.join(inc, "..", "..", "share", "libc++", "v1", "std.cppm")),
        ]:
            if os.path.isfile(candidate):
                return candidate
    raise RuntimeError("Could not locate `bits/std.cc` or `std.cppm` from Clang's default stdlib.")


def timed(label, cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  {label:60s}: {time.perf_counter() - t0:.2f}s")


def main() -> None:
    clang = os.environ.get("CXX", "clang++")
    nvcc = os.environ.get("CUDACXX", "nvcc")
    cxxflags = shlex.split(os.environ.get("CXXFLAGS", ""))
    nvccflags = shlex.split(os.environ.get("NVCCFLAGS", ""))
    cutlass_root = os.environ.get("CUTLASS_ROOT", "")
    cutlass_inc = os.environ.get(
        "CUTLASS_INCLUDE_DIR",
        os.path.join(cutlass_root, "include") if cutlass_root else "",
    )
    if not cutlass_inc or not os.path.isfile(os.path.join(cutlass_inc, "cutlass", "cutlass.h")):
        raise RuntimeError("Set CUTLASS_ROOT or CUTLASS_INCLUDE_DIR to run bench.py.")

    build_dir = os.path.join(ROOT, "build", "bench")
    os.makedirs(build_dir, exist_ok=True)
    src_dir = os.path.join(ROOT, "cutlass_gemm")
    sdk_dir = os.path.join(ROOT, "cuda_sdk")
    std_src = detect_std_module_source(clang, cxxflags)

    common = [clang, "-std=c++20", "-O3", *cxxflags, f"-I{cutlass_inc}", f"-I{src_dir}"]

    print("Precompiling `std`, `cuda.runtime`, `cutlass.cute`, `cutlass.gemm`, and `cutlass.sgemm_op`...")
    subprocess.run(
        common
        + [
            "-Wno-reserved-module-identifier",
            "-x",
            "cuda-module",
            "--no-offload-wrapper-inc",
            "--cuda-gpu-arch=sm_80",
            "--cuda-gpu-arch=sm_90",
            "--precompile",
            std_src,
            "-o",
            os.path.join(build_dir, "std.pcm"),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            "-x",
            "cuda-module",
            "--cuda-gpu-arch=sm_80",
            "--cuda-gpu-arch=sm_90",
            f"-fmodule-file=std={os.path.join(build_dir, 'std.pcm')}",
            "--precompile",
            os.path.join(sdk_dir, "cuda.runtime.cppm"),
            "-o",
            os.path.join(build_dir, "cuda.runtime.pcm"),
        ],
        check=True,
    )
    for mod, deps in [
        ("cutlass.cute", []),
        ("cutlass.gemm", ["cutlass.cute"]),
        ("cutlass.sgemm_op", ["cutlass.cute", "cutlass.gemm"]),
    ]:
        dep_flags = [f"-fmodule-file={d}={os.path.join(build_dir, d + '.pcm')}" for d in deps]
        subprocess.run(
            common
            + ["-x", "cuda-module", "--cuda-gpu-arch=sm_80", "--cuda-gpu-arch=sm_90", *dep_flags]
            + ["--precompile", os.path.join(src_dir, f"{mod}.cppm"), "-o", os.path.join(build_dir, f"{mod}.pcm")],
            check=True,
        )

    gemm_cu = os.path.join(src_dir, "basic_gemm.cu")
    cached_cu = os.path.join(src_dir, "cached_gemm.cu")
    mod_flags = [f"-fmodule-file={m}={os.path.join(build_dir, m + '.pcm')}" for m in ["std", "cutlass.cute", "cutlass.gemm"]]
    cached_flags = [
        f"-fmodule-file={m}={os.path.join(build_dir, m + '.pcm')}"
        for m in ["std", "cuda.runtime", "cutlass.cute", "cutlass.gemm", "cutlass.sgemm_op"]
    ]

    op_cppm = os.path.join(src_dir, "cutlass.sgemm_op.cppm")
    op_dep_flags = [f"-fmodule-file={m}={os.path.join(build_dir, m + '.pcm')}" for m in ["cutlass.cute", "cutlass.gemm"]]

    for title, nvcc_archs, clang_archs, tag in [
        ("Single-arch (sm_80, -O3 -c)", ["-arch=sm_80"], ["--cuda-gpu-arch=sm_80"], "80"),
        (
            "Multi-arch (sm_80 + sm_90, -O3 -c)",
            ["-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=sm_90"],
            ["--cuda-gpu-arch=sm_80", "--cuda-gpu-arch=sm_90"],
            "80_90",
        ),
    ]:
        print(f"\n=== {title} ===")
        timed(
            "nvcc textual",
            [nvcc, "-std=c++20", "-O3", *nvcc_archs, "--expt-relaxed-constexpr", "-w", *nvccflags,
             f"-I{cutlass_inc}", f"-I{src_dir}", "-DUSE_TEXTUAL_CUTLASS", "-c", gemm_cu,
             "-o", os.path.join(build_dir, f"nvcc_{tag}.o")],
        )
        timed(
            "clang textual",
            [*common, "-x", "cuda", *clang_archs, "-DUSE_TEXTUAL_CUTLASS", "-c", gemm_cu,
             "-o", os.path.join(build_dir, f"clang_textual_{tag}.o")],
        )
        timed(
            "clang + import std + cutlass.{cute,gemm} (AST in .pcm)",
            [*common, "-x", "cuda", *clang_archs, *mod_flags, "-c", gemm_cu,
             "-o", os.path.join(build_dir, f"clang_mod_{tag}.o")],
        )
        timed(
            "  [one-time] cutlass.sgemm_op.cppm -> module.o (SASS fatbin)",
            [*common, "-x", "cuda-module", *clang_archs, *op_dep_flags, "-c", op_cppm,
             "-o", os.path.join(build_dir, f"cutlass_sgemm_op_{tag}.o")],
        )
        timed(
            "clang + import std + cuda.runtime + cutlass.sgemm_op (consumer)",
            [*common, "-x", "cuda", "--no-offload-wrapper-inc", *clang_archs, *cached_flags, "-c", cached_cu,
             "-o", os.path.join(build_dir, f"clang_cached_op_{tag}.o")],
        )


if __name__ == "__main__":
    main()
