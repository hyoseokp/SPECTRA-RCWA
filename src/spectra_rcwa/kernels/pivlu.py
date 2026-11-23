"""Batched complex64 inverse with partial pivoting and no host synchronisation (CUDA only).

MAGMA's batched getrf returns to the host many times per call; here everything stays on the GPU.
Right-looking blocked LU, panel width NB, row-major [B, n, n]:

  panel kernel (one thread block per matrix): pivot search, row swap, scale, rank-1 update inside the panel
  laswp kernel: the panel's row swaps applied to the columns outside the panel
  trsm kernel:  U12 = L11^-1 A12 (unit lower NB x NB)
  A22 -= L21 U12 (cuBLAS batched GEMM on strided views)

Inverse: X = U^-1 L^-1 P, the triangular inverses by block recursion (stable without pivoting), base blocks by
cuBLAS.  :func:`inv_pivot` adds the autograd backward -X^H G X^H.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ._compile import compile_kernel

from .blockinv import block_inverse
from .cublas_inv import inv_cublas

NB = 32
TRI_BASE = 80

_SRC = r"""
__device__ __forceinline__ float2 cmul(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cinv(float2 a) { float d = a.x*a.x + a.y*a.y; return make_float2(a.x/d, -a.y/d); }

extern "C" __global__ void panel(float2* A, int* piv, int n, int k0, int nbc)
{
    float2* M = A + (long long)blockIdx.x * n * n;
    int* P = piv + (long long)blockIdx.x * n;
    __shared__ float sv[256];
    __shared__ int si[256];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int jj = 0; jj < nbc; ++jj) {
        int j = k0 + jj;
        float best = -1.f; int bi = j;
        for (int i = j + tid; i < n; i += nt) {
            float2 v = M[(long long)i * n + j];
            float m = v.x*v.x + v.y*v.y;
            if (m > best) { best = m; bi = i; }
        }
        sv[tid] = best; si[tid] = bi;
        __syncthreads();
        for (int s = nt / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            }
            __syncthreads();
        }
        int p = si[0];
        if (tid == 0) P[j] = p;
        if (p != j) {
            for (int c = k0 + tid; c < k0 + nbc; c += nt) {
                float2 t = M[(long long)j * n + c]; M[(long long)j * n + c] = M[(long long)p * n + c]; M[(long long)p * n + c] = t;
            }
        }
        __syncthreads();
        float2 ip = cinv(M[(long long)j * n + j]);
        for (int i = j + 1 + tid; i < n; i += nt) M[(long long)i * n + j] = cmul(M[(long long)i * n + j], ip);
        __syncthreads();
        int w = k0 + nbc - j - 1;
        if (w > 0) {
            long long tot = (long long)(n - j - 1) * w;
            for (long long idx = tid; idx < tot; idx += nt) {
                int i = j + 1 + (int)(idx / w), c = j + 1 + (int)(idx % w);
                M[(long long)i * n + c] = csub(M[(long long)i * n + c], cmul(M[(long long)i * n + j], M[(long long)j * n + c]));
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void laswp(float2* A, const int* piv, int n, int k0, int nbc)
{
    float2* M = A + (long long)blockIdx.y * n * n;
    const int* P = piv + (long long)blockIdx.y * n;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n || (c >= k0 && c < k0 + nbc)) return;
    for (int j = k0; j < k0 + nbc; ++j) {
        int p = P[j];
        if (p != j) { float2 t = M[(long long)j * n + c]; M[(long long)j * n + c] = M[(long long)p * n + c]; M[(long long)p * n + c] = t; }
    }
}

extern "C" __global__ void trsm_unit_lower(float2* A, int n, int k0, int nbc)
{
    // rows k0..k0+nbc-1, columns c >= k0+nbc:  U12 = L11^-1 A12 (forward substitution, one thread per column)
    float2* M = A + (long long)blockIdx.y * n * n;
    int c = k0 + nbc + blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int r = 1; r < nbc; ++r) {
        float2 acc = M[(long long)(k0 + r) * n + c];
        for (int s = 0; s < r; ++s) acc = csub(acc, cmul(M[(long long)(k0 + r) * n + k0 + s], M[(long long)(k0 + s) * n + c]));
        M[(long long)(k0 + r) * n + c] = acc;
    }
}

extern "C" __global__ void trsv(const float2* A, float2* X, int n, int k, int mode, int r0, int r1)
{
    // one thread per right-hand side, rows r0..r1 - 1.  mode 0: L x = b (unit), 1: U x = b, 2: U^H x = b, 3: L^H x = b (unit)
    long long b = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    const float2* M = A + b * n * n;
    float2* V = X + b * n * k;
    if (mode == 0) {
        for (int r = r0 + 1; r < r1; ++r) {
            float2 acc = V[(long long)r * k + c];
            for (int s = r0; s < r; ++s) acc = csub(acc, cmul(M[(long long)r * n + s], V[(long long)s * k + c]));
            V[(long long)r * k + c] = acc;
        }
    } else if (mode == 1) {
        for (int r = r1 - 1; r >= r0; --r) {
            float2 acc = V[(long long)r * k + c];
            for (int s = r + 1; s < r1; ++s) acc = csub(acc, cmul(M[(long long)r * n + s], V[(long long)s * k + c]));
            V[(long long)r * k + c] = cmul(acc, cinv(M[(long long)r * n + r]));
        }
    } else if (mode == 2) {
        for (int r = r0; r < r1; ++r) {
            float2 acc = V[(long long)r * k + c];
            for (int s = r0; s < r; ++s) { float2 u = M[(long long)s * n + r]; u.y = -u.y; acc = csub(acc, cmul(u, V[(long long)s * k + c])); }
            float2 d = M[(long long)r * n + r]; d.y = -d.y;
            V[(long long)r * k + c] = cmul(acc, cinv(d));
        }
    } else {
        for (int r = r1 - 1; r >= r0; --r) {
            float2 acc = V[(long long)r * k + c];
            for (int s = r + 1; s < r1; ++s) { float2 u = M[(long long)s * n + r]; u.y = -u.y; acc = csub(acc, cmul(u, V[(long long)s * k + c])); }
            V[(long long)r * k + c] = acc;
        }
    }
}

extern "C" __global__ void perm_from_piv(const int* piv, int* perm, int n, int nbatch)
{
    // one thread per matrix: P A = L U, perm[i] = original row now at position i
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;
    const int* P = piv + (long long)b * n; int* Q = perm + (long long)b * n;
    for (int i = 0; i < n; ++i) Q[i] = i;
    for (int j = 0; j < n; ++j) { int p = P[j]; if (p != j) { int t = Q[j]; Q[j] = Q[p]; Q[p] = t; } }
}

extern "C" __global__ void scatter_cols(const float2* Y, const int* perm, float2* X, int n)
{
    // X[b, r, perm[b, i]] = Y[b, r, i]
    long long b = blockIdx.z;
    int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    X[(b * n + r) * n + perm[b * n + i]] = Y[(b * n + r) * n + i];
}
"""
_KERNELS: dict = {}


def _k(name):
    if not _KERNELS:
        for nm in ("panel", "laswp", "trsm_unit_lower", "trsv", "perm_from_piv", "scatter_cols"):
            _KERNELS[nm] = compile_kernel(_SRC, nm)
    return _KERNELS[name]


def lu_inplace(a: torch.Tensor) -> torch.Tensor:
    """Overwrite ``a`` ([B, n, n] complex64, contiguous, CUDA) by L\\U with P a = L U; return pivots [B, n] int32."""
    b, n, _ = a.shape
    piv = torch.empty(b, n, dtype=torch.int32, device=a.device)
    for k0 in range(0, n, NB):
        nbc = min(NB, n - k0)
        _k("panel")(grid=(b, 1, 1), block=(256, 1, 1), args=[a, piv, n, k0, nbc])
        _k("laswp")(grid=((n + 127) // 128, b, 1), block=(128, 1, 1), args=[a, piv, n, k0, nbc])
        if k0 + nbc < n:
            _k("trsm_unit_lower")(grid=((n - k0 - nbc + 127) // 128, b, 1), block=(128, 1, 1), args=[a, n, k0, nbc])
            a22 = a[:, k0 + nbc:, k0 + nbc:]
            a22.baddbmm_(a[:, k0 + nbc:, k0:k0 + nbc], a[:, k0:k0 + nbc, k0 + nbc:], beta=1.0, alpha=-1.0)
    return piv


def _tri_inv(t: torch.Tensor, lower: bool) -> torch.Tensor:
    n = t.shape[-1]
    if n <= TRI_BASE:
        return inv_cublas(t)
    h = n // 2
    x11 = _tri_inv(t[:, :h, :h].contiguous(), lower)
    x22 = _tri_inv(t[:, h:, h:].contiguous(), lower)
    out = torch.zeros_like(t)
    out[:, :h, :h] = x11
    out[:, h:, h:] = x22
    if lower:
        out[:, h:, :h] = -(x22 @ (t[:, h:, :h] @ x11))
    else:
        out[:, :h, h:] = -(x11 @ (t[:, :h, h:] @ x22))
    return out


def _perm(piv):
    b, n = piv.shape
    perm = torch.empty(b, n, dtype=torch.int32, device=piv.device)
    _k("perm_from_piv")(grid=((b + 31) // 32, 1, 1), block=(32, 1, 1), args=[piv, perm, n, b])
    return perm


def lu_factor(a: torch.Tensor):
    """(L\\U, perm) with a[perm] = L U for a [B, n, n] complex64 CUDA batch."""
    lu = a.contiguous().clone()
    piv = lu_inplace(lu)
    return lu, _perm(piv).long()


TRSV_BASE = 48


def _trsv(lu, y, mode):
    """Triangular solves on the columns of y, in place: a kernel on the diagonal blocks, a GEMM with what they leave for the rest."""
    b, n, k = y.shape
    edges = list(range(0, n, TRSV_BASE)) + [n]
    chunks = list(zip(edges[:-1], edges[1:]))
    if mode in (1, 3):
        chunks.reverse()
    for r0, r1 in chunks:
        _k("trsv")(grid=((k + 7) // 8, b, 1), block=(8, 1, 1), args=[lu, y, n, k, mode, r0, r1])
        if mode == 0 and r1 < n:
            y[:, r1:].baddbmm_(lu[:, r1:, r0:r1], y[:, r0:r1], beta=1.0, alpha=-1.0)
        elif mode == 1 and r0 > 0:
            y[:, :r0].baddbmm_(lu[:, :r0, r0:r1], y[:, r0:r1], beta=1.0, alpha=-1.0)
        elif mode == 2 and r1 < n:
            y[:, r1:].baddbmm_(lu[:, r0:r1, r1:].mH, y[:, r0:r1], beta=1.0, alpha=-1.0)
        elif mode == 3 and r0 > 0:
            y[:, :r0].baddbmm_(lu[:, r0:r1, :r0].mH, y[:, r0:r1], beta=1.0, alpha=-1.0)


def lu_solve(lu, perm, x):
    """a^-1 x from :func:`lu_factor`, for a few columns."""
    y = x.gather(1, perm[:, :, None].expand(-1, -1, x.shape[-1])).contiguous()
    _trsv(lu, y, 0)
    _trsv(lu, y, 1)
    return y


def lu_solve_h(lu, perm, x):
    """a^-H x from :func:`lu_factor`, for a few columns."""
    y = x.contiguous().clone()
    _trsv(lu, y, 2)
    _trsv(lu, y, 3)
    return torch.empty_like(y).scatter_(1, perm[:, :, None].expand(-1, -1, y.shape[-1]), y)


def permuted_block_inverse(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a [B, n, n] complex64 CUDA batch: the rows that partial pivoting chooses, then the block recursion of :mod:`.blockinv` on them, faster than U^-1 L^-1 (no autograd)."""
    n = a.shape[-1]
    lu = a.contiguous().clone()
    piv = lu_inplace(lu)
    del lu
    perm = _perm(piv)
    y = block_inverse(a.gather(1, perm.long()[:, :, None].expand(-1, -1, n)))     # (P a)^-1, and a^-1 = (P a)^-1 P
    x = torch.empty_like(y)
    _k("scatter_cols")(grid=((n + 127) // 128, n, a.shape[0]), block=(128, 1, 1), args=[y, perm, x, n])
    return x


def pivoted_inverse(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a [B, n, n] complex64 CUDA batch with partial pivoting (no autograd)."""
    b, n, _ = a.shape
    lu = a.contiguous().clone()
    piv = lu_inplace(lu)
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    lo = torch.tril(lu, -1) + eye
    up = torch.triu(lu)
    y = _tri_inv(up, False) @ _tri_inv(lo, True)                  # (LU)^-1 = (P A)^-1 = A^-1 P^T
    perm = _perm(piv)
    x = torch.empty_like(y)
    _k("scatter_cols")(grid=((n + 127) // 128, n, b), block=(128, 1, 1), args=[y, perm, x, n])
    return x


class _InvPivot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        x = pivoted_inverse(a)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        xh = x.mH
        return -(xh @ (g @ xh))


def inv_pivot(a: torch.Tensor) -> torch.Tensor:
    """Differentiable sync-free pivoted inverse."""
    return _InvPivot.apply(a)
