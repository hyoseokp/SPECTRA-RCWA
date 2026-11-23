"""Runtime compilation of the kernels with NVRTC, which comes inside the torch wheel.  The kernels use only what NVRTC has built in, so no CUDA toolkit is needed; torch still asks for a CUDA home to build its include path and raises without one, so a folder without headers stands in for that call."""
from pathlib import Path

import torch


def compile_kernel(src, name, **kw):
    if not hasattr(torch.cuda, "_compile_kernel"):
        raise RuntimeError(f"spectra_rcwa: complex64 on CUDA compiles its kernels with torch.cuda._compile_kernel, which needs torch 2.8 or newer "
                           f"(found {torch.__version__}); upgrade torch, or use dtype=torch.complex128 or device='cpu'")
    import torch.utils.cpp_extension as cx
    if cx.CUDA_HOME is not None:
        return torch.cuda._compile_kernel(src, name, **kw)
    cx.CUDA_HOME = str(Path(__file__).parent)
    try:
        return torch.cuda._compile_kernel(src, name, **kw)
    finally:
        cx.CUDA_HOME = None
