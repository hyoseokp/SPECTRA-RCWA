"""The gradient from the adjoint field against autograd through the same forward pass, P, Q by structure against dense, and mirror copies of a design."""
import math

import pytest
import torch

from conftest import rel, requires_cuda, tiny_density, tiny_stack
from spectra_rcwa import DEFAULT, ColorRouterForward, Ray, shifted_disk_pupil
from spectra_rcwa.adjoint import LayerOps
from spectra_rcwa.multilayer import MultiLayerForward

WEIGHTS = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64).reshape(4, 3)
TWO = torch.stack((tiny_density(grey=False).flip(0) * 0.7 + 0.15, tiny_density()))
RECURSION, BLOCK = DEFAULT.replace(block_from_steps=10 ** 6), DEFAULT.replace(block_from_steps=1)


def _make(cfg, dtype, cls=ColorRouterForward, device="cpu", **kw):
    rays = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5, radial_order=1, azimuth_count=2)
    return cls(stack=tiny_stack(), rays=rays, wavelengths_nm=(450.0, 540.0, 640.0), device=device, dtype=dtype, config=cfg, **kw)


def _grad(fwd, rho):
    rdt = torch.float64 if fwd.cdt == torch.complex128 else torch.float32
    r = rho.to(fwd.device, rdt).requires_grad_(True)
    value = (fwd.response(r) * WEIGHTS.to(fwd.device)).sum()
    return float(value), torch.autograd.grad(value, r)[0].to(torch.float64).cpu()


CASES = [("recursion", RECURSION, {}, tiny_density()), ("block", BLOCK, {}, tiny_density()),
         ("stack, recursion", RECURSION, dict(cls=MultiLayerForward, thicknesses_um=(0.3, 0.4)), TWO), ("stack, blocks", BLOCK, dict(cls=MultiLayerForward, thicknesses_um=(0.3, 0.4)), TWO)]


@pytest.mark.parametrize("name,cfg,kw,rho", CASES, ids=[c[0] for c in CASES])
def test_adjoint_equals_autograd_complex128(name, cfg, kw, rho):
    v0, g0 = _grad(_make(cfg.replace(handwritten_backward=False), torch.complex128, **kw), rho)
    v1, g1 = _grad(_make(cfg, torch.complex128, **kw), rho)
    assert abs(v1 - v0) < 1e-12 and rel(g1, g0) < 1e-11


@pytest.mark.parametrize("name,cfg,kw,rho", CASES, ids=[c[0] for c in CASES])
def test_adjoint_complex64(name, cfg, kw, rho):
    _, g0 = _grad(_make(cfg.replace(handwritten_backward=False), torch.complex128, **kw), rho)
    _, g1 = _grad(_make(cfg, torch.complex64, **kw), rho)
    assert rel(g1, g0) < 2e-4


@pytest.mark.parametrize("cfg", [RECURSION, BLOCK], ids=["recursion", "block"])
def test_field_on_top_and_below(cfg):
    """A gradient through both outputs of the solve, and with a stored block that needs none."""
    fwd = _make(cfg, torch.complex128, cls=MultiLayerForward, thicknesses_um=(0.3, 0.4))
    plain = _make(cfg.replace(handwritten_backward=False), torch.complex128, cls=MultiLayerForward, thicknesses_um=(0.3, 0.4))
    grads = []
    for f in (fwd, plain):
        a, b = TWO[0].clone().requires_grad_(True), TWO[1].clone().requires_grad_(True)
        below, top = f.stack_field([(a, 0.3), (b, 0.4)], return_top=True)
        value = (below.abs() ** 2).sum() + 0.3 * (top.real ** 2).sum()
        grads.append(torch.autograd.grad(value, (a, b)))
    assert rel(grads[0][0], grads[1][0]) < 1e-11 and rel(grads[0][1], grads[1][1]) < 1e-11
    stored = fwd.layer_block(TWO[1], 0.4)
    grads = []
    for f in (fwd, plain):
        a = TWO[0].clone().requires_grad_(True)
        grads.append(torch.autograd.grad((f.stack_field([(a, 0.3), stored]).abs() ** 2).sum(), a)[0])
    assert rel(grads[0], grads[1]) < 1e-11


def test_stored_block_with_a_gradient():
    """On a stored block that needs a gradient autograd goes through the stack."""
    fwd = _make(DEFAULT, torch.complex128, cls=MultiLayerForward, thicknesses_um=(0.3, 0.4))
    b = TWO[1].clone().requires_grad_(True)
    g1 = torch.autograd.grad((fwd.stack_field([(TWO[0], 0.3), fwd.layer_block(b, 0.4)]).abs() ** 2).sum(), b)[0]
    b = TWO[1].clone().requires_grad_(True)
    g0 = torch.autograd.grad((fwd.stack_field([(TWO[0], 0.3), (b, 0.4)]).abs() ** 2).sum(), b)[0]
    assert rel(g1, g0) < 1e-9


def test_layer_ops_against_dense():
    fwd = _make(DEFAULT, torch.complex128)
    e_all, einv_all = fwd._econv(tiny_density()[None])
    n = len(fwd.solve_spec)
    g = fwd._geometry_pairs([(0, b) for b in range(n)], [fwd.stack.rayleigh_nudge] * n)
    p, q = fwd._pq(e_all, einv_all, g)
    x = torch.randn(n, p.shape[-1], 3, dtype=p.dtype, generator=torch.Generator().manual_seed(1))
    y = torch.randn(n, 3, p.shape[-1], dtype=p.dtype, generator=torch.Generator().manual_seed(2))
    dense = LayerOps(e_all[g.lam_index], einv_all[g.lam_index], g.kx, g.ky)
    with torch.no_grad():
        structured = LayerOps(e_all[g.lam_index], einv_all[g.lam_index], g.kx, g.ky)
        for ops in (dense, structured):
            assert rel(ops.q_left(x), q @ x) < 1e-12 and rel(ops.p_right(y), y @ p) < 1e-12 and rel(ops.pq(), p @ q) < 1e-12


def test_switch_to_the_route_before_the_adjoint():
    from spectra_rcwa import AUTOGRAD_ROUTE
    v0, g0 = _grad(_make(AUTOGRAD_ROUTE, torch.complex128), tiny_density())
    v1, g1 = _grad(_make(DEFAULT, torch.complex128), tiny_density())
    assert abs(v1 - v0) < 1e-12 and rel(g1, g0) < 1e-11


@requires_cuda
def test_structured_kernels_against_torch():
    from spectra_rcwa.geometry import BDD
    from spectra_rcwa.kernels.bdd_mul import bdd_left, bdd_right
    from spectra_rcwa.kernels.thin import thin_mn
    b, n = 3, 40
    dev = "cuda"
    x = torch.randn(b, 2 * n, 2 * n, dtype=torch.complex64, device=dev)
    m, m2 = [BDD(*[torch.randn(b, n, dtype=torch.complex64, device=dev) for _ in range(4)]) for _ in range(2)]
    eye = torch.eye(2 * n, dtype=x.dtype, device=dev)
    with torch.no_grad():
        assert rel(bdd_left(m, x, -1, 1), eye - m.left(x)) < 1e-6 and rel(bdd_right(m, x, -1, 1), eye - m.right(x)) < 1e-6
        e, s, q, g = [torch.randn_like(x) for _ in range(4)]
        ss, qq, gg = m.right(s), m2.left(q), m.right(m2.left(g))
        mm, kk = thin_mn(e, s, q, g, m, m2)
        assert rel(mm, eye + 0.5 * (e + gg - ss - qq)) < 1e-6 and rel(kk, 0.5 * (e - gg - ss + qq)) < 1e-6


@requires_cuda
def test_lu_solves_against_torch():
    from spectra_rcwa.kernels.pivlu import lu_factor, lu_solve, lu_solve_h
    for b, n in ((1, 130), (20, 130)):
        a = torch.randn(b, n, n, dtype=torch.complex64, device="cuda")
        x = torch.randn(b, n, 2, dtype=torch.complex64, device="cuda")
        with torch.no_grad():
            lu, perm = lu_factor(a)
            s, sh = lu_solve(lu, perm, x), lu_solve_h(lu, perm, x)
        assert rel(a.cdouble() @ s.cdouble(), x.cdouble()) < 1e-2 and rel(a.mH.cdouble() @ sh.cdouble(), x.cdouble()) < 1e-2


@requires_cuda
@pytest.mark.parametrize("name,cfg,kw,rho", CASES, ids=[c[0] for c in CASES])
def test_adjoint_on_the_gpu(name, cfg, kw, rho):
    v0, g0 = _grad(_make(cfg.replace(handwritten_backward=False), torch.complex128, **kw), rho)
    v1, g1 = _grad(_make(cfg, torch.complex64, device="cuda", **kw), rho)
    assert abs(v1 - v0) < 2e-5 * abs(v0) and rel(g1, g0) < 3e-3


def _mirror_cases(rho, rays):
    """(name, density, rays, wells of the copy in the order of the original) for the exact mirrors of the lattice: i -> N - i, and x <-> y."""
    n = rho.shape[-1]
    flip = (n - torch.arange(n)) % n
    return [("mirror x", rho[..., flip, :], [Ray(r.theta_rad, math.pi - r.phi_rad, r.weight) for r in rays], [1, 0, 3, 2]),
            ("mirror y", rho[..., :, flip], [Ray(r.theta_rad, -r.phi_rad, r.weight) for r in rays], [2, 3, 0, 1]),
            ("swap x y", rho.transpose(-2, -1), [Ray(r.theta_rad, math.pi / 2 - r.phi_rad, r.weight) for r in rays], [0, 2, 1, 3])]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("name,cfg,kw,rho", CASES, ids=[c[0] for c in CASES])
def test_mirror_copies(device, name, cfg, kw, rho):
    """A mirror of the lattice maps the design, the rays and the wells onto each other, whatever the inverses and the batch: 48 solves in one batch."""
    rays = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5, radial_order=2, azimuth_count=8)
    rho32 = rho.float().to(device)

    def tabs(r, ry):
        cls = kw.get("cls", ColorRouterForward)
        extra = {k: v for k, v in kw.items() if k != "cls"}
        return cls(stack=tiny_stack(), rays=ry, wavelengths_nm=(450.0, 540.0, 640.0), device=device, dtype=torch.complex64, config=cfg, **extra).response(r)

    with torch.no_grad():
        base = tabs(rho32, rays)
        assert base.shape == (4, 3)
        for _, r, ry, wells in _mirror_cases(rho32, rays):
            assert rel(tabs(r, ry)[wells], base) < 5e-5
