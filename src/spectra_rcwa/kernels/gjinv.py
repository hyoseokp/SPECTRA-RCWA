"""Batched complex64 inverse of small matrices (n <= MAX_N) by Gauss-Jordan with partial pivoting, one CUDA block per matrix, in shared memory.

Faster than cuBLAS getrf + getri at the sizes the block recursion ends in, and one launch instead of two library calls.
"""
from __future__ import annotations

import torch

from ._compile import compile_kernel

MAX_N = 75                                                 # 48 KB of shared memory
THREADS = 256

_SRC = r"""
__device__ __forceinline__ float2 cm(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
extern "C" __global__ void gj_inverse(const float2* __restrict__ In, float2* __restrict__ Out, int n)
{
    extern __shared__ float2 S[];                          // n x n matrix, then the column of factors
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ int piv[76];
    const int tid = threadIdx.x, nt = blockDim.x, nn = n * n;
    float2* F = S + nn;
    const float2* A = In + (long long)blockIdx.x * nn;
    float2* O = Out + (long long)blockIdx.x * nn;
    for (int q = tid; q < nn; q += nt) S[q] = A[q];
    __syncthreads();
    for (int k = 0; k < n; ++k) {
        float best = -1.f; int bi = k;
        for (int i = k + tid; i < n; i += nt) { float2 v = S[i * n + k]; float m = v.x*v.x + v.y*v.y; if (m > best) { best = m; bi = i; } }
        sv[tid] = best; si[tid] = bi;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            __syncthreads();
        }
        const int p = si[0];
        if (tid == 0) piv[k] = p;
        if (p != k)
            for (int c = tid; c < n; c += nt) { float2 t = S[k * n + c]; S[k * n + c] = S[p * n + c]; S[p * n + c] = t; }
        __syncthreads();
        float2 d = S[k * n + k];
        float m = d.x*d.x + d.y*d.y;
        float2 w = make_float2(d.x / m, -d.y / m);
        for (int i = tid; i < n; i += nt) F[i] = S[i * n + k];
        __syncthreads();
        for (int c = tid; c < n; c += nt) S[k * n + c] = c == k ? w : cm(w, S[k * n + c]);
        __syncthreads();
        for (int q = tid; q < nn; q += nt) {
            int i = q / n, c = q - i * n;
            if (i == k) continue;
            float2 f = F[i], r = S[k * n + c], v = c == k ? make_float2(0.f, 0.f) : S[q];
            S[q] = make_float2(v.x - (f.x*r.x - f.y*r.y), v.y - (f.x*r.y + f.y*r.x));
        }
        __syncthreads();
    }
    for (int k = n - 1; k >= 0; --k) {                     // the row swaps, undone on the columns
        const int p = piv[k];
        if (p == k) continue;
        for (int i = tid; i < n; i += nt) { float2 t = S[i * n + k]; S[i * n + k] = S[i * n + p]; S[i * n + p] = t; }
        __syncthreads();
    }
    for (int q = tid; q < nn; q += nt) O[q] = S[q];
}
"""
_KERNEL = None


def gj_inverse(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a [B, n, n] complex64 CUDA batch, n <= MAX_N (no autograd)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel(_SRC, "gj_inverse")
    b, n, _ = a.shape
    a = a.contiguous()
    out = torch.empty_like(a)
    _KERNEL(grid=(b, 1, 1), block=(THREADS, 1, 1), shared_mem=(n * n + n) * 8, args=[a, out, n])
    return out
