"""Eigendecomposition-free batched RCWA forward and adjoint for freeform metasurfaces and stacks of them (PyTorch)."""
from .config import AUTOGRAD_ROUTE, DEFAULT, REFERENCE_C128, SolverConfig
from .forward import ColorRouterForward, Ray, Stack, value_and_grad
from .matfun import cos_sinc, expm_taylor, gamma, sqrtm_db
from .boundary import boundary_direct, layer_smatrix
from .multilayer import MultiLayerForward
from .pupil import shifted_disk_pupil

__version__ = "1.3.1"

__all__ = [
    "ColorRouterForward", "AUTOGRAD_ROUTE",
    "DEFAULT", "MultiLayerForward", "REFERENCE_C128", "Ray", "SolverConfig", "Stack", "boundary_direct", "cos_sinc",
    "expm_taylor", "gamma", "layer_smatrix", "shifted_disk_pupil", "sqrtm_db", "value_and_grad",
]
