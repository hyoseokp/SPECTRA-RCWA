"""cuSOLVER for one matrix or a few large ones (complex64, CUDA).

The batched kernels of this package work on one matrix per thread block, so that a single matrix leaves the GPU idle: the LU of one 7442 x 7442 matrix
takes 566 ms on an RTX 5090, and 53 ms in cuSOLVER, which takes a batch one matrix after the other.
"""
from __future__ import annotations

import torch

from ..lapack import cusolver

BLOCK_UP_TO = 2500                                         # above it the block recursion inverts a single matrix faster


def few(m: torch.Tensor) -> bool:
    """Whether cuSOLVER factorises the batch [B, n, n] faster than the batched kernels."""
    return m.shape[0] <= max(1, m.shape[-1] // 500)


def inverse(m: torch.Tensor) -> torch.Tensor:
    with cusolver():
        return torch.linalg.inv_ex(m)[0]


def lu_factor(m: torch.Tensor):
    with cusolver():
        return torch.linalg.lu_factor_ex(m)[:2]


def lu_solve(lu, pivots, x: torch.Tensor, adjoint: bool = False) -> torch.Tensor:
    with cusolver():
        return torch.linalg.lu_solve(lu, pivots, x, adjoint=adjoint)
