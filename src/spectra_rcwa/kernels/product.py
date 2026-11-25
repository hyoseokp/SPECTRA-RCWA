"""Complex products on CUDA as three real ones, where that is faster.

(a + ib)(c + id) = (ac - bd) + i((a + b)(c + d) - ac - bd).  cuBLAS's single real GEMM runs much nearer to the peak than its complex one, and three
replace four; for a batch of smaller matrices the batched complex GEMM is faster.  On an RTX 5090 one product takes 0.57 of the time at n = 2178 and
0.42 at 7442, on an RTX 3060 0.9; the relative error of a product stays near 1e-6.  In complex128 the multiplications are the bottleneck of a GeForce at any
size, and three for four save 8 to 21 % from n = 250 on.
"""
from __future__ import annotations

import torch

ONE_FROM, BATCH_FROM = 1800, 2000                          # complex64: smallest dimension from which it is used, for one product and for a batch
DOUBLE_FROM = 250                                          # complex128, on a GPU with slow double precision
_SLOW_DOUBLE: dict = {}


def _slow_double(device) -> bool:
    if device not in _SLOW_DOUBLE:
        _SLOW_DOUBLE[device] = "GeForce" in torch.cuda.get_device_name(device)
    return _SLOW_DOUBLE[device]


def _as_three(a, b) -> bool:
    if not a.is_cuda or (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad)):
        return False
    smallest = min(a.shape[1], a.shape[2], b.shape[2])
    if a.dtype == torch.complex128:
        return smallest >= DOUBLE_FROM and _slow_double(a.device)
    return a.dtype == torch.complex64 and smallest >= (ONE_FROM if a.shape[0] == 1 else BATCH_FROM)


def _three_real(a, b):
    ar, ai, br, bi = a.real.contiguous(), a.imag.contiguous(), b.real.contiguous(), b.imag.contiguous()
    p, q = ar @ br, ai @ bi
    ar += ai
    br += bi
    s = ar @ br
    s -= p
    s -= q
    p -= q
    return torch.complex(p, s)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b for [B, m, k] and [B, k, n]."""
    if not _as_three(a, b):
        return a @ b
    if a.dtype == torch.complex128:
        return _three_real(a, b)
    if a.shape[0] == 1:                                    # the unbatched real GEMM is the fast one
        return _three_real(a[0], b[0])[None]
    return torch.stack([_three_real(x, y) for x, y in zip(a, b)])


def baddbmm(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, beta: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """beta c + alpha a @ b."""
    if _as_three(a, b):
        p = matmul(a, b)
        return alpha * p if beta == 0.0 else torch.add(p if alpha == 1.0 else alpha * p, c, alpha=beta)
    return torch.baddbmm(c, a, b, beta=beta, alpha=alpha)
