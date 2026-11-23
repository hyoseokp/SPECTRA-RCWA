"""Direct boundary solve: against the S-matrix route and the eigendecomposition route, its adjoint, and
layer composition (one layer of thickness d equals two layers of thickness d/2)."""
import torch

from conftest import eig_transmitted, layer_inputs, rel, tiny_density, tiny_forward
from spectra_rcwa import REFERENCE_C128
from spectra_rcwa import matfun
from spectra_rcwa.boundary import boundary_direct, boundary_direct_reference, layer_smatrix, transmitted_smatrix


def _setup(eps_in=1.0):
    fwd = tiny_forward(eps_in=eps_in)
    p, a, g = layer_inputs(fwd, tiny_density())
    gam = matfun.gamma(a, REFERENCE_C128)
    e = matfun.expm_taylor((1j * g.omega[:, None, None] * g.thickness) * gam, REFERENCE_C128)
    return p, a, g, gam, e


def test_direct_equals_smatrix_and_eig():
    for eps_in in (1.0, 1.46 ** 2):                                  # air and glass superstrate
        p, a, g, gam, e = _setup(eps_in)
        t_direct = boundary_direct(gam, e, g.vi.right(p), g.vo.right(p), g.e_in)
        assert rel(t_direct, transmitted_smatrix(p, gam, e, g)) < 1e-11
        assert rel(t_direct, eig_transmitted(p, a, g)) < 1e-11


def test_direct_reference_formula():
    p, a, g, gam, e = _setup()
    ki, ko = g.vi.right(p), g.vo.right(p)
    assert rel(boundary_direct(gam, e, ki, ko, g.e_in), boundary_direct_reference(gam, e, ki, ko, g.e_in)) < 1e-12


def test_adjoint_matches_autograd():
    p, a, g, gam, e = _setup()
    ins = [x.detach()[:3].clone().requires_grad_(True) for x in (gam, e, g.vi.right(p), g.vo.right(p))]
    j = g.e_in[:3]
    gt = torch.randn(3, gam.shape[-1], 2, dtype=gam.dtype, generator=torch.Generator().manual_seed(4))
    g_hand = torch.autograd.grad(boundary_direct(*ins, j), ins, gt)
    g_auto = torch.autograd.grad(boundary_direct_reference(*ins, j), ins, gt)
    for u, v in zip(g_hand, g_auto):
        assert rel(u, v) < 1e-10


def test_adjoint_gradcheck():
    gen = torch.Generator().manual_seed(5)
    n, r = 6, 2
    eye = torch.eye(n, dtype=torch.complex128)
    ins = [(eye * s + 0.3 * torch.randn(1, n, n, dtype=torch.complex128, generator=gen)).requires_grad_(True)
           for s in (2.0, 0.5, 1.0, 1.5)]
    j = torch.randn(1, n, r, dtype=torch.complex128, generator=gen)
    assert torch.autograd.gradcheck(lambda *x: boundary_direct(*x, j), ins)


def test_one_layer_equals_two_half_layers():
    p, a, g, gam, _ = _setup()
    w = 1j * g.omega[:, None, None] * g.thickness
    e_full = matfun.expm_taylor(w * gam, REFERENCE_C128)
    e_half = matfun.expm_taylor(0.5 * w * gam, REFERENCE_C128)
    t, r = layer_smatrix(p, gam, e_full, g.vf_inv)
    th, rh = layer_smatrix(p, gam, e_half, g.vf_inv)
    eye = torch.eye(gam.shape[-1], dtype=gam.dtype)
    inner = torch.linalg.inv(eye - rh @ rh)                          # Redheffer product of two identical halves
    assert rel(th @ inner @ th, t) < 1e-10
    assert rel(rh + th @ inner @ rh @ th, r) < 1e-10
