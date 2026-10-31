#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build reports/raw reports/figures
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo --ptxas-options=-v \
  --shared -Xcompiler -fPIC csrc/library.cu -o build/libllm_cuda_lab.so 2> reports/raw/build.log
nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo csrc/native_check.cu \
  -Lbuild -lllm_cuda_lab -Xlinker -rpath -Xlinker '$ORIGIN' -o build/native_check
