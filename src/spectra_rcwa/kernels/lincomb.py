"""Batched complex64 linear combination in one NVRTC-compiled CUDA kernel, with autograd.

    out[b] = sum_i coef[b, i] * X_i[b] + diag[b] * I        (X_i: [B, n, n], up to 4 inputs)

replaces a chain of torch elementwise ops (each a full read/write pass over B x n x n) by one pass.
The backward is the same kernel: grad X_i = conj(coef[:, i]) * grad_out (torch's complex convention).
On CPU, or for complex128, :func:`lincomb` evaluates the same expression with plain torch ops.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ._compile import compile_kernel

_SRC = r"""
extern "C" __global__ void lincomb(const float2* __restrict__ x0, const float2* __restrict__ x1,
                                   const float2* __restrict__ x2, const float2* __restrict__ x3,
                                   const float2* __restrict__ coef, const float2* __restrict__ diag,
                                   float2* __restrict__ out, int k, int n, int per_i, int total_i, int use_diag)
{
    // the launcher passes Python ints as 32-bit C ints: take them as int, widen here
    long long per = per_i, total = total_i;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        long long b = idx / per;
        long long r = idx - b * per;
        const float2* c = coef + b * 4;
        float2 v = make_float2(0.f, 0.f);
        float2 x, cc;
        x = x0[idx]; cc = c[0]; v.x += cc.x * x.x - cc.y * x.y; v.y += cc.x * x.y + cc.y * x.x;
        if (k > 1) { x = x1[idx]; cc = c[1]; v.x += cc.x * x.x - cc.y * x.y; v.y += cc.x * x.y + cc.y * x.x; }
        if (k > 2) { x = x2[idx]; cc = c[2]; v.x += cc.x * x.x - cc.y * x.y; v.y += cc.x * x.y + cc.y * x.x; }
        if (k > 3) { x = x3[idx]; cc = c[3]; v.x += cc.x * x.x - cc.y * x.y; v.y += cc.x * x.y + cc.y * x.x; }
        if (use_diag) {
            long long row = r / n, col = r - row * n;
            if (row == col) { float2 d = diag[b]; v.x += d.x; v.y += d.y; }
        }
        out[idx] = v;
    }
}
"""
_KERNEL = None
ENABLED = True       # set False to force the plain-torch path everywhere


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel(_SRC, "lincomb", nvcc_options=["--use_fast_math"])
    return _KERNEL


def _launch(xs, coef, diag):
    b, n, _ = xs[0].shape
    out = torch.empty(xs[0].shape, dtype=xs[0].dtype, device=xs[0].device)
    per = n * n
    total = b * per
    xs_c = [x.contiguous() for x in xs] + [xs[0]] * (4 - len(xs))
    threads = 256
    blocks = min((total + threads - 1) // threads, 65535 * 4)
    _kernel()(grid=(blocks, 1, 1), block=(threads, 1, 1),
              args=[*xs_c, coef.contiguous(), diag.contiguous() if diag is not None else coef, out,
                    len(xs), n, per, total, int(diag is not None)])
    return out


class _LinComb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coef, diag, *xs):
        ctx.save_for_backward(coef)
        ctx.k = len(xs)
        return _launch(xs, coef, diag)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (coef,) = ctx.saved_tensors
        g = g.contiguous()
        grads = []
        for i in range(ctx.k):
            if ctx.needs_input_grad[2 + i]:
                c = torch.zeros_like(coef)
                c[:, 0] = coef[:, i].conj()
                grads.append(_launch([g], c, None))
            else:
                grads.append(None)
        return (None, None, *grads)


def lincomb(xs, coefs, diag=None):
    """``sum_i coefs[i] * xs[i] (+ diag * I)`` for a batch of square matrices.

    ``coefs`` and ``diag``: Python scalars or [B] tensors (treated as constants, no gradient).
    """
    b = xs[0].shape[0]
    dev = xs[0].device
    use_kernel = ENABLED and xs[0].dtype == torch.complex64 and xs[0].is_cuda and len(xs) <= 4
    if not use_kernel:
        out = sum((c.detach().reshape(-1, 1, 1) if torch.is_tensor(c) else c) * x for c, x in zip(coefs, xs))
        if diag is not None:
            d = diag.detach().reshape(-1, 1, 1) if torch.is_tensor(diag) else diag
            out = out + d * torch.eye(xs[0].shape[-1], dtype=xs[0].dtype, device=dev)
        return out
    coef = torch.zeros(b, 4, dtype=torch.complex64, device=dev)
    for i, c in enumerate(coefs):
        coef[:, i] = c.detach().reshape(-1) if torch.is_tensor(c) else c
    d = None
    if diag is not None:
        d = (diag.detach().reshape(-1) if torch.is_tensor(diag)
             else torch.full((b,), complex(diag), dtype=torch.complex64, device=dev)).to(torch.complex64)
    return _LinComb.apply(coef, d, *xs)
