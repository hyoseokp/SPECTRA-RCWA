"""The elementwise parts of products with P and Q, in single CUDA passes (complex64, no autograd).  With x = [x1; x2] in blocks of N rows:

    proj_rows:  w = a x1 + b x2
    lift_rows:  [out1; out2] = s [c z + d u1; e z + f u2]

and the same on columns.  a, b, c, e are real [B, N]; s a complex [B]; d, f are +1 or -1.  A product with P is proj, a GEMM with E^-1, lift; one with Q is proj, two GEMMs with E, lift.
"""
from __future__ import annotations

import torch

from ._compile import compile_kernel

_SRC = r"""
__device__ __forceinline__ float2 cm(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 rs(float a, float2 x) { return make_float2(a*x.x, a*x.y); }
__device__ __forceinline__ float2 ad(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
extern "C" __global__ void proj_rows(const float2* __restrict__ X, const float* __restrict__ a, const float* __restrict__ b, float2* __restrict__ W, int N, int m, int total_i)
{
    long long total = total_i, per = (long long)N * m;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / m, c = rem - r * m;
        const float2* x = X + bb * 2 * per;
        W[idx] = ad(rs(a[bb * N + r], x[(long long)r * m + c]), rs(b[bb * N + r], x[(long long)(r + N) * m + c]));
    }
}
extern "C" __global__ void lift_rows(const float2* __restrict__ Z, const float2* __restrict__ U1, const float2* __restrict__ U2, const float* __restrict__ c1, const float* __restrict__ c2,
                                     const float2* __restrict__ s, float2* __restrict__ O, int N, int m, int per_u_i, int d_i, int f_i, int total_i)
{
    long long total = total_i, per = (long long)N * m, per_u = per_u_i;
    float d = (float)d_i, f = (float)f_i;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / m, c = rem - r * m;
        float2 z = Z[idx];
        float2 sc = s[bb];
        O[bb * 2 * per + (long long)r * m + c] = cm(sc, ad(rs(c1[bb * N + r], z), rs(d, U1[bb * per_u + rem])));
        O[bb * 2 * per + (long long)(r + N) * m + c] = cm(sc, ad(rs(c2[bb * N + r], z), rs(f, U2[bb * per_u + rem])));
    }
}
extern "C" __global__ void proj_cols(const float2* __restrict__ X, const float* __restrict__ a, const float* __restrict__ b, float2* __restrict__ W, int N, int m, int total_i)
{
    // X [B, m, 2N], W [B, m, N]
    long long total = total_i, per = (long long)N * m;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / N, c = rem - r * N;
        const float2* x = X + bb * 2 * per + (long long)r * 2 * N;
        W[idx] = ad(rs(a[bb * N + c], x[c]), rs(b[bb * N + c], x[c + N]));
    }
}
extern "C" __global__ void lift_cols(const float2* __restrict__ Z, const float2* __restrict__ X, const float* __restrict__ c1, const float* __restrict__ c2, float2* __restrict__ O, int N, int m, int d_i, int f_i, int total_i)
{
    // [out1, out2] = [z c1 + d x2, z c2 + f x1]; Z [B, m, N], X, O [B, m, 2N]
    long long total = total_i, per = (long long)N * m;
    float d = (float)d_i, f = (float)f_i;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / N, c = rem - r * N;
        float2 z = Z[idx];
        const float2* x = X + bb * 2 * per + (long long)r * 2 * N;
        float2* o = O + bb * 2 * per + (long long)r * 2 * N;
        o[c] = ad(rs(c1[bb * N + c], z), rs(d, x[c + N]));
        o[c + N] = ad(rs(c2[bb * N + c], z), rs(f, x[c]));
    }
}
"""
_KERNELS: dict = {}


def _k(name):
    if not _KERNELS:
        for nm in ("proj_rows", "lift_rows", "proj_cols", "lift_cols"):
            _KERNELS[nm] = compile_kernel(_SRC, nm, nvcc_options=["--use_fast_math"])
    return _KERNELS[name]


def _grid(total):
    return (min((total + 255) // 256, 65535 * 4), 1, 1)


def proj_rows(x, a, b):
    bs, rows, m = x.shape
    n = rows // 2
    w = torch.empty(bs, n, m, dtype=x.dtype, device=x.device)
    total = bs * n * m
    _k("proj_rows")(grid=_grid(total), block=(256, 1, 1), args=[x.contiguous(), a.contiguous(), b.contiguous(), w, n, m, total])
    return w


def lift_rows(z, u1, u2, c1, c2, s, d, f, per_u):
    """``u1``, ``u2``: [B, N, m] views, each a row block of a tensor with ``per_u`` elements per batch entry."""
    bs, n, m = z.shape
    o = torch.empty(bs, 2 * n, m, dtype=z.dtype, device=z.device)
    total = bs * n * m
    _k("lift_rows")(grid=_grid(total), block=(256, 1, 1), args=[z, u1, u2, c1.contiguous(), c2.contiguous(), s, o, n, m, per_u, int(d), int(f), total])
    return o


def proj_cols(x, a, b):
    bs, m, cols = x.shape
    n = cols // 2
    w = torch.empty(bs, m, n, dtype=x.dtype, device=x.device)
    total = bs * n * m
    _k("proj_cols")(grid=_grid(total), block=(256, 1, 1), args=[x.contiguous(), a.contiguous(), b.contiguous(), w, n, m, total])
    return w


def lift_cols(z, x, c1, c2, d, f):
    bs, m, n = z.shape
    o = torch.empty_like(x)
    total = bs * n * m
    _k("lift_cols")(grid=_grid(total), block=(256, 1, 1), args=[z.contiguous(), x.contiguous(), c1.contiguous(), c2.contiguous(), o, n, m, int(d), int(f), total])
    return o
