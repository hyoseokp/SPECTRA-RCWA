"""Batched inverse by block recursion (Schur complement); all work in batched GEMMs.

    inv([[A, B], [C, D]]):  X = inv(A),  T = C X,  S = D - T B,  Si = inv(S),  U = X B
                            [[X + U Si T, -U Si], [-Si T, Si]]

recursing on A and S down to ``BASE``, where a Gauss-Jordan kernel (:mod:`.gjinv`, pivoted inside the base block) or,
on CPU, torch.linalg.inv takes over.  There is no pivoting across blocks: use it only on matrices that do not
need it (the later Denman-Beavers iterates and the boundary-solve matrices; not the first DB iterates).
No host synchronisation.  :func:`inv_block` adds the autograd backward -X^H G X^H.
"""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .gjinv import MAX_N, gj_inverse
from .product import baddbmm, matmul

BASE = MAX_N


def _base_inv(a):
    if a.is_cuda and a.dtype == torch.complex64:
        return gj_inverse(a)
    return torch.linalg.inv(a)


def block_inverse(a: torch.Tensor) -> torch.Tensor:
    """Inverse of [B, n, n] (no autograd)."""
    n = a.shape[-1]
    if n <= BASE:
        return _base_inv(a)
    h = n // 2
    a11, a12 = a[:, :h, :h], a[:, :h, h:]                          # strided views, no copies
    a21, a22 = a[:, h:, :h], a[:, h:, h:]
    x = block_inverse(a11)
    t = matmul(a21, x)
    s = baddbmm(a22, t, a12, beta=1.0, alpha=-1.0)
    si = block_inverse(s)
    u = matmul(x, a12)
    b12 = -matmul(u, si)
    b21 = -matmul(si, t)
    b11 = baddbmm(x, b12, t, beta=1.0, alpha=-1.0)           # X + U Si T = X - B12 T
    out = torch.empty_like(a)
    out[:, :h, :h] = b11
    out[:, :h, h:] = b12
    out[:, h:, :h] = b21
    out[:, h:, h:] = si
    return out


class _InvBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        x = block_inverse(a.contiguous())
        ctx.save_for_backward(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        xh = x.mH
        return -(xh @ (g @ xh))


def inv_block(a: torch.Tensor) -> torch.Tensor:
    """Differentiable block-recursive inverse."""
    return _InvBlock.apply(a)
