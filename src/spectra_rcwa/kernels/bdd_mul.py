"""Products with a block-diagonal-diagonal matrix, and I minus such a product, in one CUDA pass (complex64, no autograd).

    out = shift I + scale B x,      out = shift I + scale x B

Each reads x once and writes out once, against the two splits, four elementwise products, two sums and a concatenation of the same expression in torch.
"""
from __future__ import annotations

import torch

from ._compile import compile_kernel

_SRC = r"""
__device__ __forceinline__ float2 cm(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
// v: [B, 4, N] = (a, b, c, d)
extern "C" __global__ void bdd_left(const float2* __restrict__ X, const float2* __restrict__ v, float2* __restrict__ O, int N, int k, int per_i, int total_i, int scale_i, int shift_i)
{
    long long per = per_i, total = total_i;
    int n2 = 2 * N;
    float scale = (float)scale_i, shift = (float)shift_i;       // the launcher passes Python ints as 32-bit C ints
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / k, c = rem - r * k;
        int rb = r >= N, rl = rb ? r - N : r;
        const float2* V = v + bb * 4 * N;
        long long base = bb * per;
        float2 p = cm(V[(rb ? 2 : 0) * N + rl], X[base + (long long)rl * k + c]);
        float2 q = cm(V[(rb ? 3 : 1) * N + rl], X[base + (long long)(rl + N) * k + c]);
        float2 o = make_float2(scale * (p.x + q.x), scale * (p.y + q.y));
        if (r == c) o.x += shift;
        O[idx] = o;
    }
}
extern "C" __global__ void bdd_right(const float2* __restrict__ X, const float2* __restrict__ v, float2* __restrict__ O, int N, int k, int per_i, int total_i, int scale_i, int shift_i)
{
    long long per = per_i, total = total_i;
    int n2 = 2 * N;
    float scale = (float)scale_i, shift = (float)shift_i;       // the launcher passes Python ints as 32-bit C ints
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / n2, c = rem - r * n2;
        int cb = c >= N, cl = cb ? c - N : c;
        const float2* V = v + bb * 4 * N;
        long long row = bb * per + (long long)r * n2;
        float2 p = cm(X[row + cl], V[(cb ? 1 : 0) * N + cl]);
        float2 q = cm(X[row + cl + N], V[(cb ? 3 : 2) * N + cl]);
        float2 o = make_float2(scale * (p.x + q.x), scale * (p.y + q.y));
        if (r == c) o.x += shift;
        O[idx] = o;
    }
}
"""
_KERNELS: dict = {}


def _k(name):
    if not _KERNELS:
        for nm in ("bdd_left", "bdd_right"):
            _KERNELS[nm] = compile_kernel(_SRC, nm, nvcc_options=["--use_fast_math"])
    return _KERNELS[name]


def _run(name, x, m, scale, shift):
    b, rows, cols = x.shape
    n = m.a.shape[-1]
    v = torch.stack((m.a, m.b, m.c, m.d), 1).to(x.dtype).contiguous()
    out = torch.empty_like(x)
    per, total = rows * cols, b * rows * cols
    _k(name)(grid=(min((total + 255) // 256, 65535 * 4), 1, 1), block=(256, 1, 1), args=[x.contiguous(), v, out, n, cols, per, total, int(scale), int(shift)])
    return out


def _fused(x):
    return x.is_cuda and x.dtype == torch.complex64 and not torch.is_grad_enabled()


def bdd_left(m, x, scale=1, shift=0):
    """shift I + scale m @ x for scale and shift whole numbers; ``shift`` only for a square x."""
    if _fused(x):
        return _run("bdd_left", x, m, scale, shift)
    out = scale * m.left(x)
    return out + shift * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) if shift else out


def bdd_right(m, x, scale=1, shift=0):
    """shift I + scale x @ m for scale and shift whole numbers."""
    if _fused(x):
        return _run("bdd_right", x, m, scale, shift)
    out = scale * m.right(x)
    return out + shift * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device) if shift else out
