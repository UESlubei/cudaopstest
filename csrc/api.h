#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
extern "C" {
int lab_rms(const half*, const half*, const half*, half*, half*, int, int, float, int, cudaStream_t);
int lab_gemm(const half*, const half*, half*, int, int, int, int, cudaStream_t);
int lab_decode(const half*, const half*, const half*, const int*, half*, float*, int, int, int, int, int, int, int, cudaStream_t);
const char* lab_error(int);
}
