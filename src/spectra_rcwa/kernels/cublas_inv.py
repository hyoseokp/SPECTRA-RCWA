"""Batched complex64 inverse through cuBLAS getrfBatched + getriBatched (ctypes, on torch's own cuBLAS handle).

No host synchronisation (the info array stays on the device).  Row-major in, row-major out: cuBLAS sees A^T
(column-major) and returns (A^T)^-1 = (A^-1)^T, i.e. A^-1 in row-major.

The cuBLAS library is the one torch loaded.  Set ``SPECTRA_RCWA_CUBLAS`` to its path if it is not found
automatically (torch/lib on Windows wheels, nvidia/cublas/lib on Linux wheels).
"""
from __future__ import annotations

import ctypes
import glob
import os

import torch

_LIB = None


def _candidates():
    env = os.environ.get("SPECTRA_RCWA_CUBLAS")
    if env:
        yield env
    dirs = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    try:
        import nvidia.cublas as nvc                                       # Linux pip wheels
        for p in nvc.__path__:
            dirs += [os.path.join(p, "lib"), os.path.join(p, "bin")]
    except ImportError:
        pass
    for d in dirs:
        for pat in ("cublas64_*.dll", "libcublas.so*"):
            yield from sorted(glob.glob(os.path.join(d, pat)))


def _cublas():
    global _LIB
    if _LIB is None:
        for path in _candidates():
            try:
                _LIB = ctypes.CDLL(path)
                break
            except OSError:
                continue
        if _LIB is None:
            raise RuntimeError("cuBLAS library not found; set SPECTRA_RCWA_CUBLAS")
        for name in ("cublasCgetrfBatched", "cublasCgetriBatched"):
            getattr(_LIB, name).restype = ctypes.c_int
    return _LIB


def inv_cublas(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a [B, n, n] complex64 CUDA batch (no autograd)."""
    b, n, _ = a.shape
    lu = a.contiguous().clone()
    out = torch.empty_like(lu)
    piv = torch.empty(b, n, dtype=torch.int32, device=a.device)
    info = torch.empty(b, dtype=torch.int32, device=a.device)
    stride = n * n * lu.element_size()
    pa = torch.arange(b, dtype=torch.int64, device=a.device) * stride + lu.data_ptr()
    pc = torch.arange(b, dtype=torch.int64, device=a.device) * stride + out.data_ptr()
    h = ctypes.c_void_p(torch.cuda.current_blas_handle())
    lib = _cublas()
    st = lib.cublasCgetrfBatched(h, n, ctypes.c_void_p(pa.data_ptr()), n, ctypes.c_void_p(piv.data_ptr()),
                                 ctypes.c_void_p(info.data_ptr()), b)
    if st != 0:
        raise RuntimeError(f"cublasCgetrfBatched status {st}")
    st = lib.cublasCgetriBatched(h, n, ctypes.c_void_p(pa.data_ptr()), n, ctypes.c_void_p(piv.data_ptr()),
                                 ctypes.c_void_p(pc.data_ptr()), n, ctypes.c_void_p(info.data_ptr()), b)
    if st != 0:
        raise RuntimeError(f"cublasCgetriBatched status {st}")
    return out
