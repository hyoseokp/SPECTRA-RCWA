"""K_i = X Y_i and K_o = X Y_o in one CUDA pass, with backward (complex64).

Y_i and Y_o are block-diagonal-diagonal ([[a, b], [c, d]] with length-N diagonal blocks), so

    (X Y)[r, c]     = X[r, c] a[c] + X[r, c+N] c[c]      (c < N)
    (X Y)[r, N + c] = X[r, c] b[c] + X[r, c+N] d[c]

and both products read X once.  :func:`bdd_pair` falls back to ``BDD.right`` on CPU / complex128.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ._compile import compile_kernel

_SRC = r"""
__device__ __forceinline__ float2 cm(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 cmc(float2 a, float2 b) { return make_float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y); }  // a * conj(b)
__device__ __forceinline__ float2 ad(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
// v: [B, 8, N] = (ai, bi, ci, di, ao, bo, co, do)
extern "C" __global__ void bdd_pair(const float2* __restrict__ X, const float2* __restrict__ v, float2* __restrict__ Ki,
                                    float2* __restrict__ Ko, int N, int total)
{
    int n2 = 2 * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int bb = idx / (n2 * N), rem = idx - bb * n2 * N;
        int r = rem / N, c = rem - r * N;
        long long base = ((long long)bb * n2 + r) * n2;
        const float2* V = v + (long long)bb * 8 * N;
        float2 x1 = X[base + c], x2 = X[base + c + N];
        Ki[base + c]     = ad(cm(x1, V[0*N + c]), cm(x2, V[2*N + c]));
        Ki[base + N + c] = ad(cm(x1, V[1*N + c]), cm(x2, V[3*N + c]));
        Ko[base + c]     = ad(cm(x1, V[4*N + c]), cm(x2, V[6*N + c]));
        Ko[base + N + c] = ad(cm(x1, V[5*N + c]), cm(x2, V[7*N + c]));
    }
}
extern "C" __global__ void bdd_pair_bwd(const float2* __restrict__ Gi, const float2* __restrict__ Go, const float2* __restrict__ v,
                                        float2* __restrict__ GX, int N, int total)
{
    int n2 = 2 * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int bb = idx / (n2 * N), rem = idx - bb * n2 * N;
        int r = rem / N, c = rem - r * N;
        long long base = ((long long)bb * n2 + r) * n2;
        const float2* V = v + (long long)bb * 8 * N;
        float2 gi1 = Gi[base + c], gi2 = Gi[base + N + c], go1 = Go[base + c], go2 = Go[base + N + c];
        GX[base + c]     = ad(ad(cmc(gi1, V[0*N + c]), cmc(gi2, V[1*N + c])), ad(cmc(go1, V[4*N + c]), cmc(go2, V[5*N + c])));
        GX[base + N + c] = ad(ad(cmc(gi1, V[2*N + c]), cmc(gi2, V[3*N + c])), ad(cmc(go1, V[6*N + c]), cmc(go2, V[7*N + c])));
    }
}
"""
_KERNELS: dict = {}


def _k(name):
    if not _KERNELS:
        for nm in ("bdd_pair", "bdd_pair_bwd"):
            _KERNELS[nm] = compile_kernel(_SRC, nm)
    return _KERNELS[name]


def _grid(total):
    return (min((total + 255) // 256, 65535 * 4), 1, 1)


class _BDDPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v):
        x = x.contiguous()
        b, n2, _ = x.shape
        n = n2 // 2
        ki, ko = torch.empty_like(x), torch.empty_like(x)
        total = b * n2 * n
        _k("bdd_pair")(grid=_grid(total), block=(256, 1, 1), args=[x, v, ki, ko, n, total])
        ctx.save_for_backward(v)
        return ki, ko

    @staticmethod
    @once_differentiable
    def backward(ctx, gi, go):
        (v,) = ctx.saved_tensors
        gi = torch.zeros_like(go) if gi is None else gi.contiguous()
        go = torch.zeros_like(gi) if go is None else go.contiguous()
        b, n2, _ = gi.shape
        n = n2 // 2
        gx = torch.empty_like(gi)
        total = b * n2 * n
        _k("bdd_pair_bwd")(grid=_grid(total), block=(256, 1, 1), args=[gi, go, v, gx, n, total])
        return gx, None


def bdd_pair(x, yi, yo, fused: bool = True):
    """``(x @ yi, x @ yo)`` for block-diagonal-diagonal ``yi``, ``yo`` (:class:`spectra_rcwa.geometry.BDD`)."""
    if not (fused and x.dtype == torch.complex64 and x.is_cuda):
        return yi.right(x), yo.right(x)
    v = torch.stack([yi.a, yi.b, yi.c, yi.d, yo.a, yo.b, yo.c, yo.d], 1).to(torch.complex64).contiguous()
    return _BDDPair.apply(x, v)
