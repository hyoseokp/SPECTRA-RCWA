"""The direction across the edges of a pattern, as a smooth field, for the "normal" rule of the in-plane permittivity (SolverConfig.factorization).

Where the permittivity jumps, the field along the edge and the displacement across it are continuous, so a truncated Fourier series has to multiply the
one by [[eps]] and the other by [[1/eps]]^-1 (Li; Popov and Neviere; Schuster et al., J. Opt. Soc. Am. A 24, 2880 (2007)):

    D = [[eps]] E - ([[eps]] - [[1/eps]]^-1) n (n . E)

The n here is not a vector but the tensor n n^T, as the smoothed structure tensor of the density divided by its trace.  It is n n^T on a straight edge
and falls to I / 2 where edges of all directions are equally far, so it needs neither a sign nor a direction inside the features, where it does not act.
"""
from __future__ import annotations

import math

import torch

LENGTH = 0.045                                             # decay length of the smoothing, in the shorter period
PRE = 0.006                                                # width of the edges the gradient is taken of, in the shorter period (at least one sample)


def normal_tensor(rho: torch.Tensor, period, b) -> torch.Tensor:
    """[D, 3, H, W] (xx, yy, xy), float64 and differentiable, of densities [D, H, W] on a cell with reciprocal lattice vectors ``b``."""
    rho = rho.to(torch.float64)
    h, w = rho.shape[-2:]
    (b1x, b1y), (_, b2y) = b
    m = torch.fft.fftfreq(h, d=1.0 / h, dtype=torch.float64, device=rho.device)[:, None]
    n = torch.fft.fftfreq(w, d=1.0 / w, dtype=torch.float64, device=rho.device)[None, :]
    gx, gy = 2 * math.pi * m * b1x, 2 * math.pi * (m * b1y + n * b2y)
    g2 = gx ** 2 + gy ** 2
    short = min(period)
    pre = max(PRE * short, max(period) / max(h, w))
    f = torch.fft.fft2(rho) * torch.exp(-0.5 * pre ** 2 * g2)
    dx, dy = torch.fft.ifft2(1j * gx * f).real, torch.fft.ifft2(1j * gy * f).real
    j = torch.fft.ifft2(torch.fft.fft2(torch.stack((dx * dx, dy * dy, dx * dy), 1)) / (1 + (LENGTH * short) ** 2 * g2)).real
    trace = j[:, :1] + j[:, 1:2]
    floor = 1e-9 * trace.detach().amax((2, 3), keepdim=True) + 1e-30
    half = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64, device=rho.device)[None, :, None, None]
    return (j + floor * half) / (trace + floor)            # a uniform layer gets I / 2


def in_plane(e, inv, tensor):
    """[..., 3, N, N] (xx, yy, xy = yx) of the rule from [[eps]], [[1/eps]]^-1 and the Toeplitz matrices [..., 3, N, N] of the tensor."""
    jump = (e - inv)[..., None, :, :]
    part = 0.5 * (jump @ tensor + tensor @ jump)
    zero = torch.zeros_like(e)
    return torch.stack((e, e, zero), -3) - part
