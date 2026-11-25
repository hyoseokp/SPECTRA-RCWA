"""Inverses and linear solves of torch for a batch of matrices.

After torch.set_num_threads with more than one thread, torch with MKL on Windows (2.6.0 and 2.10.0) returns wrong inverses,
or raises, from the LU factorization of a CPU batch of matrices of 150 and more.  One matrix at a time it is right, and the threads then
run inside each factorization.  Whether set_num_threads was called cannot be seen, so a CPU batch of matrices of SPLIT_FROM
and more is taken one matrix at a time whenever torch has more than one thread; smaller matrices, one thread and CUDA go
to torch as they are.  On CUDA the inverses run in cuSOLVER, except for a batch of complex128 matrices, where MAGMA is faster.
"""
from __future__ import annotations

import contextlib

import torch

SPLIT_FROM = 128


@contextlib.contextmanager
def cusolver(use: bool = True):
    previous = torch.backends.cuda.preferred_linalg_library() if use else None
    if use:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    try:
        yield
    finally:
        if use:
            torch.backends.cuda.preferred_linalg_library(previous)


def _library(a: torch.Tensor):
    return cusolver(a.is_cuda and (a.dtype == torch.complex64 or a.dim() < 3 or a.shape[0] <= 4))


def _split(a: torch.Tensor) -> bool:
    return a.dim() > 2 and a.shape[-1] >= SPLIT_FROM and not a.is_cuda and torch.get_num_threads() > 1


def _each(fn, a, *rest):
    """fn on one matrix of a [..., n, n] at a time, with the right-hand sides ``rest`` [..., n, k] of the same batch."""
    rest = [r.expand(*a.shape[:-2], -1, -1).flatten(0, -3) for r in rest]
    out = torch.stack([fn(*m) for m in zip(a.flatten(0, -3), *rest)])
    return out.reshape(*a.shape[:-2], *out.shape[-2:])


def _inv_ex(a):
    return torch.linalg.inv_ex(a)[0]


def inv(a: torch.Tensor) -> torch.Tensor:
    """torch.linalg.inv."""
    with _library(a):
        return _each(torch.linalg.inv, a) if _split(a) else torch.linalg.inv(a)


def inv_unchecked(a: torch.Tensor) -> torch.Tensor:
    """The inverse of torch.linalg.inv_ex: no check for a singular matrix, and on CUDA no synchronisation with the host."""
    with _library(a):
        return _each(_inv_ex, a) if _split(a) else _inv_ex(a)


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.linalg.solve for right-hand sides b [..., n, k]."""
    return _each(torch.linalg.solve, a, b) if _split(a) else torch.linalg.solve(a, b)
