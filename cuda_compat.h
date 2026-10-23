// CUDA 13 compatibility shim for the RWKV-7 kernels.
//
// CUDA 13 removed the built-in `atomicAdd(float2*, float2)` overload that
// cuda/rwkv7_cmix_bf16_v5.cu, cuda/rwkv7_tmix_mix6_bf16_v5.cu and
// cuda/rwkv7_tmix_kk_pre_bf16_v5.cu rely on.  mtp.py force-includes this header
// into every .cu translation unit (via `-include`), so the vendor kernel
// sources stay untouched.
//
// Only compiled by nvcc (`__CUDACC__`); harmless if ever seen by a host pass.

#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)

__device__ __forceinline__ float2 atomicAdd(float2* address, float2 val) {
    float2 old;
    old.x = atomicAdd(&address->x, val.x);
    old.y = atomicAdd(&address->y, val.y);
    return old;
}

#endif  // __CUDACC__ && __CUDA_ARCH__ < 900
