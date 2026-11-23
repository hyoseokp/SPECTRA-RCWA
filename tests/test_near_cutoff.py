"""A layer order at cutoff: A has an eigenvalue near 0, and P (or Q) a small singular value.

Gamma = A^(1/2) is not analytic there.  The solves are tuned so that the smallest eigenvalue of A is 1e-6: the route through
Gamma with 4 Denman-Beavers steps then has a complex64 gradient wrong by a factor up to 900, and a complex128 gradient wrong
by a factor up to 9.  The default route has no square root and needs no special handling.  Exactly at lambda = 1e-6 even
converged complex128 through Gamma has a relative gradient error of about 1e-3, so the reference is that route at lambda =
+-3e-5, checked against central differences and interpolated in wavelength (value and gradient are smooth through the cutoff;
the two gradients differ by about 1e-4).
"""
import math

import pytest
import torch

from conftest import tiny_density, tiny_stack
from spectra_rcwa import DEFAULT, REFERENCE_C128, ColorRouterForward, Ray
from spectra_rcwa.multilayer import MultiLayerForward

WL_P = 561.0226740980095                                             # P nearly singular (TM-like cutoff)
WL_Q = 567.2608138143514                                             # Q nearly singular (TE-like cutoff)
WL_FAR = 551.1807493141757                                           # smallest |eigenvalue| 0.06
NEIGHBOURS = {WL_P: (561.0176984444099, 561.0279928325349),          # lambda = +3e-5, -3e-5
              WL_Q: (567.2554973898364, 567.2664968209413),
              WL_FAR: (WL_FAR, WL_FAR)}
WEIGHTS = torch.tensor([[1.0], [-0.6], [0.4], [0.2]], dtype=torch.float64)
RAY = (Ray(math.radians(5.0), math.radians(22.5), 1.0),)
REF = REFERENCE_C128.replace(boundary="direct")


def _forward(wl, config=REF, dtype=torch.complex128, cls=ColorRouterForward, **kw):
    return cls(stack=tiny_stack(), rays=RAY, wavelengths_nm=(wl,), device="cpu", dtype=dtype, config=config, **kw)


def _value_grad(fwd, rho, weights=WEIGHTS):
    rdt = torch.float64 if fwd.cdt == torch.complex128 else torch.float32
    r = rho.to(rdt).detach().clone().requires_grad_(True)
    value = (fwd.response(r) * weights).sum()
    return float(value), torch.autograd.grad(value, r)[0].to(torch.float64)


def _reference(wl, rho, weights=WEIGHTS, **kw):
    """Value and gradient at wl from complex128 on both sides of the cutoff."""
    w1, w2 = NEIGHBOURS[wl]
    s = 0.0 if w1 == w2 else (wl - w1) / (w2 - w1)
    v1, g1 = _value_grad(_forward(w1, **kw), rho, weights)
    v2, g2 = _value_grad(_forward(w2, **kw), rho, weights)
    return (1 - s) * v1 + s * v2, (1 - s) * g1 + s * g2


def _l2(a, b):
    return float((a - b).norm() / b.norm())


def test_smallest_eigenvalue_of_the_tuned_wavelengths():
    rho = tiny_density()
    for wl, lam in ((WL_P, 1e-6), (WL_Q, 1e-6), (NEIGHBOURS[WL_P][0], 3e-5), (NEIGHBOURS[WL_Q][1], 3e-5)):
        fwd = _forward(wl)
        e_all, einv_all = fwd._econv(rho[None])
        _, a = fwd._layer(e_all, einv_all, fwd._geometry_pairs([(0, 0)], [fwd.stack.rayleigh_nudge]))
        assert abs(float(torch.linalg.eigvals(a).abs().min()) - lam) < 0.02 * lam


@pytest.mark.parametrize("wl", [NEIGHBOURS[WL_P][0], NEIGHBOURS[WL_P][1], NEIGHBOURS[WL_Q][0], NEIGHBOURS[WL_Q][1]])
def test_reference_against_central_differences(wl):
    rho = tiny_density()
    fwd = _forward(wl)
    grad = _value_grad(fwd, rho)[1]
    gen = torch.Generator().manual_seed(5)
    for _ in range(2):
        v = torch.randn(rho.shape, dtype=torch.float64, generator=gen)
        v = v / v.norm()
        h = 1e-4
        with torch.no_grad():
            fd = float((((fwd.response(rho + h * v) - fwd.response(rho - h * v)) * WEIGHTS).sum()) / (2 * h))
        assert abs(float((grad * v).sum()) - fd) < 1e-4 * abs(fd)


@pytest.mark.parametrize("wl", [WL_P, WL_Q])
def test_default_complex64_at_cutoff(wl):
    rho = tiny_density()
    v_ref, g_ref = _reference(wl, rho)
    v, g = _value_grad(_forward(wl, DEFAULT, torch.complex64), rho)
    assert abs(v - v_ref) < 2e-5 * abs(v_ref)
    assert _l2(g, g_ref) < 3e-3


@pytest.mark.parametrize("wl", [WL_P, WL_Q])
def test_default_complex128_at_cutoff(wl):
    rho = tiny_density()
    v_ref, g_ref = _reference(wl, rho)
    v, g = _value_grad(_forward(wl, DEFAULT), rho)
    assert abs(v - v_ref) < 2e-7 * abs(v_ref)                         # the reference is interpolated
    assert _l2(g, g_ref) < 2e-5


def test_default_gradient_at_cutoff_against_central_differences():
    rho = tiny_density()
    fwd = _forward(WL_P, DEFAULT)
    grad = _value_grad(fwd, rho)[1]
    v = torch.randn(rho.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    v = v / v.norm()
    h = 1e-5
    with torch.no_grad():
        fd = float((((fwd.response(rho + h * v) - fwd.response(rho - h * v)) * WEIGHTS).sum()) / (2 * h))
    assert abs(float((grad * v).sum()) - fd) < 1e-8 * abs(fd) + 1e-10


def test_batch_with_one_solve_at_cutoff():
    rho = tiny_density()
    w = torch.tensor([[1.0, -0.3], [-0.6, 0.5], [0.4, 0.2], [0.2, -0.7]], dtype=torch.float64)
    v0, g0 = 0.0, 0.0
    for li, wl in enumerate((WL_P, WL_FAR)):
        vi, gi = _reference(wl, rho, w[:, li:li + 1])
        v0, g0 = v0 + vi, g0 + gi
    mix = ColorRouterForward(stack=tiny_stack(), rays=RAY, wavelengths_nm=(WL_P, WL_FAR), device="cpu", dtype=torch.complex64, config=DEFAULT)
    r = rho.float().detach().clone().requires_grad_(True)
    value = (mix.response(r) * w).sum()
    g1 = torch.autograd.grad(value, r)[0].to(torch.float64)
    assert abs(float(value) - v0) < 2e-5 * abs(v0) and _l2(g1, g0) < 3e-3


@pytest.mark.parametrize("wl", [WL_P, WL_Q])
def test_block_route_complex64_at_cutoff(wl):
    rho = tiny_density()
    v_ref, g_ref = _reference(wl, rho)
    v, g = _value_grad(_forward(wl, DEFAULT.replace(block_from_steps=1), torch.complex64), rho)
    assert abs(v - v_ref) < 1e-4 * abs(v_ref)
    assert _l2(g, g_ref) < 1e-2


def test_stack_with_one_layer_at_cutoff():
    rho = torch.stack((tiny_density(grey=False).flip(0) * 0.7 + 0.15, tiny_density()))
    kw = dict(cls=MultiLayerForward, thicknesses_um=(0.3, 0.6))
    v_ref, g_ref = _reference(WL_P, rho, **kw)
    v, g = _value_grad(_forward(WL_P, DEFAULT, torch.complex64, **kw), rho)
    assert abs(v - v_ref) < 2e-5 * abs(v_ref)
    assert _l2(g, g_ref) < 3e-3
