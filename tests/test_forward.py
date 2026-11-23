"""Whole forward: response against the eigendecomposition route, complex64 accuracy, and a complex128
central finite-difference check of the gradient."""
import math

import torch

from conftest import eig_transmitted, rel, tiny_density, tiny_forward
from spectra_rcwa import DEFAULT, REFERENCE_C128, shifted_disk_pupil


def _eig_response(fwd, rho):
    fwd._transmitted = lambda p, a, g: eig_transmitted(p, a, g)
    return fwd.response(rho)


def test_response_matches_eig():
    rho = tiny_density()
    ref = _eig_response(tiny_forward(), rho)
    assert rel(tiny_forward(config=REFERENCE_C128).response(rho), ref) < 1e-11
    assert rel(tiny_forward(config=DEFAULT).response(rho), ref) < 1e-6
    assert rel(tiny_forward(dtype=torch.complex64, config=DEFAULT).response(rho.float()), ref) < 1e-4


def test_batched_densities_are_independent():
    fwd = tiny_forward(config=DEFAULT)
    r1, r2 = tiny_density(), tiny_density(grey=False)
    both = fwd.responses(torch.stack((r1, r2)))
    assert rel(both[0], fwd.response(r1)) < 1e-12 and rel(both[1], fwd.response(r2)) < 1e-12


def test_gradient_central_differences_c128():
    weights = torch.tensor([[1.0, 0.2, -0.5], [0.3, -1.0, 0.4], [-0.2, 0.6, 1.0], [0.5, 0.1, -0.3]],
                           dtype=torch.float64)
    gen = torch.Generator().manual_seed(7)
    for cfg in (DEFAULT, REFERENCE_C128, REFERENCE_C128.replace(boundary="direct")):
        fwd = tiny_forward(config=cfg)
        obj = lambda r: (fwd.response(r) * weights).sum()            # noqa: E731
        rho = tiny_density().requires_grad_(True)
        grad = torch.autograd.grad(obj(rho), rho)[0]
        for _ in range(3):
            v = torch.randn(rho.shape, dtype=torch.float64, generator=gen)
            v = v / v.norm()
            h = 1e-5
            with torch.no_grad():
                fd = float((obj(rho + h * v) - obj(rho - h * v)) / (2 * h))
            assert abs(float((grad * v).sum()) - fd) < 1e-7 * abs(fd) + 1e-10


def test_pupil_weights():
    rays = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5)
    assert len(rays) == 16
    assert abs(sum(r.weight for r in rays) - math.cos(math.radians(2.5)) ** 4) < 1e-14
