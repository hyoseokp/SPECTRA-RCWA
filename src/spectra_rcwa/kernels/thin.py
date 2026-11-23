"""The two matrices of a thin step in the wave basis of a reference admittance, in one CUDA pass (complex64).

With S = i S P, Q = i Q S, G = Q G P and E = C - I from the entire functions of a step, Y0 and Z0 = Y0^-1 block-diagonal-diagonal:

    M = I + (E + Z0 G Y0) / 2 - (S Y0 + Z0 Q) / 2,   N = (E - Z0 G Y0) / 2 - (S Y0 - Z0 Q) / 2

(T = M^-1, R = N T).  One pass reads each of the four inputs once, against some twenty read and write passes of the same expression in torch.
"""
from __future__ import annotations

import torch

from ._compile import compile_kernel

_SRC = r"""
__device__ __forceinline__ float2 cm(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 ad(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
// y, z: [B, 4, N] = (a, b, c, d)
extern "C" __global__ void thin_mn(const float2* __restrict__ E, const float2* __restrict__ S, const float2* __restrict__ Q, const float2* __restrict__ G,
                                   const float2* __restrict__ y, const float2* __restrict__ z, float2* __restrict__ M, float2* __restrict__ K, int N, int per_i, int total_i)
{
    long long per = per_i, total = total_i;
    int n2 = 2 * N;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        long long bb = idx / per;
        int rem = (int)(idx - bb * per);
        int r = rem / n2, c = rem - r * n2;
        int rb = r >= N, cb = c >= N, rl = rb ? r - N : r, cl = cb ? c - N : c;
        const float2* Y = y + bb * 4 * N; const float2* Z = z + bb * 4 * N;
        long long base = bb * per;
        // columns: X Y = X[:, cl] (a or b)[cl] + X[:, cl + N] (c or d)[cl]; rows: Z X = (a or c)[rl] X[rl, :] + (b or d)[rl] X[rl + N, :]
        float2 y1 = Y[(cb ? 1 : 0) * N + cl], y2 = Y[(cb ? 3 : 2) * N + cl];
        float2 z1 = Z[(rb ? 2 : 0) * N + rl], z2 = Z[(rb ? 3 : 1) * N + rl];
        long long r1 = base + (long long)rl * n2, r2 = base + (long long)(rl + N) * n2, rr = base + (long long)r * n2;
        float2 ss = ad(cm(S[rr + cl], y1), cm(S[rr + cl + N], y2));
        float2 qq = ad(cm(z1, Q[r1 + c]), cm(z2, Q[r2 + c]));
        float2 g1 = ad(cm(G[r1 + cl], y1), cm(G[r1 + cl + N], y2));
        float2 g2 = ad(cm(G[r2 + cl], y1), cm(G[r2 + cl + N], y2));
        float2 gg = ad(cm(z1, g1), cm(z2, g2));
        float2 e = E[rr + c];
        float2 m = make_float2(0.5f * (e.x + gg.x - ss.x - qq.x), 0.5f * (e.y + gg.y - ss.y - qq.y));
        if (r == c) m.x += 1.0f;
        M[idx] = m;
        K[idx] = make_float2(0.5f * (e.x - gg.x - ss.x + qq.x), 0.5f * (e.y - gg.y - ss.y + qq.y));
    }
}
"""
_KERNEL = None


def thin_mn(e, s, q, g, y0, z0):
    """(M, N) above for [B, 2N, 2N] complex64 CUDA tensors and block-diagonal-diagonal y0, z0 (no autograd)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel(_SRC, "thin_mn", nvcc_options=["--use_fast_math"])
    b, n2, _ = e.shape
    y = torch.stack((y0.a, y0.b, y0.c, y0.d), 1).to(e.dtype).contiguous()
    z = torch.stack((z0.a, z0.b, z0.c, z0.d), 1).to(e.dtype).contiguous()
    m, k = torch.empty_like(e), torch.empty_like(e)
    total = b * n2 * n2
    _KERNEL(grid=(min((total + 255) // 256, 65535 * 4), 1, 1), block=(256, 1, 1),
            args=[e.contiguous(), s.contiguous(), q.contiguous(), g.contiguous(), y, z, m, k, n2 // 2, n2 * n2, total])
    return m, k
