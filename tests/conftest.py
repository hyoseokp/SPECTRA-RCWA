import pytest
import torch

from spectra_rcwa import REFERENCE_C128, ColorRouterForward, Stack, shifted_disk_pupil


def tiny_density(shape=(16, 16), grey=True):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1]), indexing="ij")
    rho = ((xx - 0.2) ** 2 + (yy + 0.1) ** 2 < 0.45).double()
    rho[: shape[0] // 3, shape[1] // 2:] = 1.0
    return 0.1 + 0.8 * rho if grey else rho


def tiny_stack(eps_in=1.0, orders=(2, 2)):
    return Stack(period_um=(1.2, 1.2), thickness_um=0.6, density_shape=(16, 16), fourier_orders=orders,
                 eps_design=lambda wl: (2.0 + 0.01 / wl ** 2) ** 2, eps_fill=1.0, eps_in=eps_in, eps_out=1.46 ** 2,
                 detector_z_um=0.9, detector_samples=(8, 8))


def tiny_forward(dtype=torch.complex128, config=REFERENCE_C128, eps_in=1.0, device="cpu", orders=(2, 2)):
    rays = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5,
                              radial_order=1, azimuth_count=2)
    return ColorRouterForward(stack=tiny_stack(eps_in, orders), rays=rays, wavelengths_nm=(450.0, 540.0, 640.0),
                              device=device, dtype=dtype, config=config)


def layer_inputs(fwd, density):
    """P, A = PQ and the geometry of every solve of ``fwd`` for one density."""
    e_all, einv_all = fwd._econv(density[None].to(fwd.device))
    n = len(fwd.solve_spec)
    g = fwd._geometry_pairs([(0, b) for b in range(n)], [fwd.stack.rayleigh_nudge] * n)
    p, a = fwd._layer(e_all, einv_all, g)
    return p, a, g


def eig_transmitted(p, a, g):
    """Classical RCWA with an eigendecomposition (the formulation the matrix functions replace)."""
    n2 = a.shape[-1]
    vals, x = torch.linalg.eig(a)
    kz = torch.sqrt(vals)
    kz = torch.where(torch.imag(kz) < 0, -kz, kz)
    h = torch.linalg.solve(p, x * kz[:, None, :])
    w = g.vf_inv.left(h)
    a_blk, b_blk = x + w, x - w
    phase = torch.exp(1j * g.omega[:, None] * kz * g.thickness)
    b_phi = b_blk * phase[:, None, :]
    inv_p, inv_m = torch.linalg.inv(a_blk + b_phi), torch.linalg.inv(a_blk - b_phi)
    sig, dif = inv_p + inv_m, inv_p - inv_m
    eye = torch.eye(n2, dtype=a.dtype)
    lay11 = x @ (phase[:, :, None] * sig + dif)
    lay21 = x @ (sig + phase[:, :, None] * dif) - eye
    r1 = eye - g.sin12.left(lay21)
    rhs = torch.cat((g.sin11.left(g.e_in), g.sin12.left(lay11)), -1)
    sol = torch.linalg.solve(r1, rhs)
    s1v = lay11 @ sol[..., :2]
    s12p = lay21 + lay11 @ sol[..., 2:]
    r2 = eye - g.sout21.right(s12p)
    return g.sout11.left(torch.linalg.solve(r2, s1v))


def eig_gamma(a):
    """A^(1/2) from eigenpairs, scalar branch e^(i pi/4) sqrt(-i lambda) (kz > 0 or Im kz > 0)."""
    vals, x = torch.linalg.eig(a)
    kz = complex(2 ** -0.5, 2 ** -0.5) * torch.sqrt(-1j * vals)
    return x @ torch.diag_embed(kz) @ torch.linalg.inv(x)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
