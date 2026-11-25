#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${DIR}/build}"

cmake -S "${DIR}" -B "${BUILD_DIR}" -G Ninja "$@"
cmake --build "${BUILD_DIR}"
