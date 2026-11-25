"""The "normal" rule of a crossed grating: gradient, energy, convergence, lattices, exact shapes; and wells with circular orders."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from conftest import tiny_stack
from spectra_rcwa import AUTOGRAD_ROUTE, DEFAULT, ColorRouterForward, Ray, Stack
from spectra_rcwa import problem as P
from spectra_rcwa.normals import normal_tensor
from spectra_rcwa.geometry import reciprocal

NORMAL = DEFAULT.replace(factorization="normal")
STACK = Stack(period_um=(0.7, 0.6), thickness_um=0.3, density_shape=(33, 33), fourier_orders=(3, 3), eps_design=9.0 + 0.1j, eps_out=2.1)
LATTICES = [dict(), dict(truncation="circular"), dict(lattice_angle_deg=65.0)]


def blob(n=33):
    x = torch.arange(n, dtype=torch.float64) / n
    return torch.sigmoid(8 * (torch.cos(2 * math.pi * x)[:, None] + 0.6 * torch.sin(2 * math.pi * x)[None, :] + 0.4 * torch.cos(2 * math.pi * (x[:, None] + x[None, :]))))


def make(stack, config=NORMAL, **kw):
    kw = dict(device="cpu", dtype=torch.complex128) | kw
    return ColorRouterForward(stack=stack, rays=(Ray(0.2, 0.4, 0.6), Ray(0.0, 0.0, 0.4)), wavelengths_nm=(500.0, 600.0), config=config, **kw)


def objective(fwd, r):
    t, q = fwd.efficiencies([(r, 0.3), (1 - r, 1.1)])
    w = torch.cos(torch.arange(t.numel(), dtype=torch.float64)).reshape(t.shape)
    return (t.double().cpu() * w).sum() + (q.double().cpu() * w).sum()


@pytest.mark.parametrize("lattice", LATTICES)
def test_gradient(lattice):
    stack = dataclasses.replace(STACK, **lattice)
    grads = []
    for cfg in (NORMAL, AUTOGRAD_ROUTE.replace(factorization="normal")):
        rho = blob().requires_grad_(True)
        grads.append(torch.autograd.grad(objective(make(stack, cfg), rho), rho)[0])
    norm = grads[1].norm()
    assert float((grads[0] - grads[1]).norm() / norm) < 1e-10
    d, h = grads[0] / norm, 1e-5
    assert math.isclose(float((objective(make(stack), blob() + h * d) - objective(make(stack), blob() - h * d)) / (2 * h)), float(norm), rel_tol=1e-7)


@pytest.mark.parametrize("lattice", LATTICES)
def test_energy(lattice):
    fwd = make(dataclasses.replace(STACK, eps_design=9.0, **lattice))
    t, r = fwd.efficiencies([(blob(), 0.3)])
    assert float((t.sum(1) + r.sum(1) - 1).abs().max()) < 1e-11


def test_tensor():
    rho = blob(64)[None]
    t = normal_tensor(rho, (0.7, 0.6), reciprocal((0.7, 0.6)))[0]
    assert float((t[0] + t[1] - 1).abs().max()) < 1e-12 and float((t[0] * t[1] - t[2] ** 2).min()) > -1e-9      # trace 1, positive
    bars = torch.zeros(1, 64, 64, dtype=torch.float64)
    bars[:, :32] = 1.0
    t = normal_tensor(bars, (1.0, 1.0), reciprocal((1.0, 1.0)))[0]
    assert float((t[0] - 1).abs().max()) < 1e-6 and float(t[1:].abs().max()) < 1e-6
    flat = normal_tensor(torch.full((1, 8, 8), 0.3), (1.0, 1.0), reciprocal((1.0, 1.0)))[0]
    assert torch.allclose(flat[0], torch.full((8, 8), 0.5, dtype=torch.float64), atol=1e-6)


def pillar(**kw):
    return dict(period_um=[0.5, 0.5], fourier_orders=[4, 4], truncation="circular", wavelengths_nm=[633.0], incidence=[dict(theta_deg=12.0, phi_deg=23.0)], eps_in=1.0, eps_out=2.13,
                eps_design=12.0, eps_fill=1.0, rayleigh_nudge=0.0, device="cpu", precision="double", grid=[256, 256],
                layers=[dict(thickness_um=0.6, pattern=dict(type="disk", radius_um=0.15))]) | kw


def box(problem):
    return P._embed(P.solve(problem), [10, 10])


def test_converges_faster_than_laurent():
    reference = box(pillar(factorization="normal", fourier_orders=[9, 9]))
    error = {rule: np.abs(box(pillar(factorization=rule, fourier_orders=[5, 5])) - reference).max() for rule in ("laurent", "normal")}
    large = np.abs(box(pillar(factorization="laurent", fourier_orders=[9, 9])) - reference).max()
    assert error["normal"] < 0.25 * error["laurent"] and large < 0.7 * error["laurent"]        # and Laurent moves towards the same value


def test_exact_disk_is_the_limit_of_the_samples(monkeypatch):
    problem = pillar(factorization="normal", fourier_orders=[3, 3], grid=[512, 512])
    exact = box(problem)
    monkeypatch.setattr(P, "_exact", lambda *a: False)
    assert np.abs(box(problem) - exact).max() < 2e-3


def test_the_same_hexagonal_lattice_in_another_basis():
    a, b = (P.solve(pillar(factorization="normal", lattice_angle_deg=angle, truncation="auto", layers=[dict(thickness_um=0.6, pattern=dict(type="disk", radius_um=0.15, center_um=[0.31, 0.12]))]))["arrays"]
            for angle in (60.0, 120.0))
    value = {(int(m), int(n)): a["T"][0, 0, i] for i, (m, n) in enumerate(a["orders"])}
    assert max(np.abs(value[int(m), int(n + m)] - b["T"][0, 0, i]).max() for i, (m, n) in enumerate(b["orders"])) < 1e-8


def test_pixel_gradient():
    rho = np.random.default_rng(0).random((6, 5))
    problem = pillar(factorization="normal", grid=[33, 29], fourier_orders=[3, 3], layers=[dict(thickness_um=0.3, pattern=dict(type="array", density=rho.tolist(), pixels=True))])
    out = P.gradient(problem, order=(0, 0), fd_check=True)
    assert math.isclose(out["fd_check"]["central_difference"], out["fd_check"]["gradient_norm"], rel_tol=1e-6)


def test_same_as_li_for_a_1d_grating_and_as_laurent_for_a_film():
    lam = dict(P.example("lamellar_grating"), device="cpu", precision="double")
    assert np.array_equal(P.solve(lam)["arrays"]["T"], P.solve(dict(lam, factorization="normal"))["arrays"]["T"])
    slab = dict(P.example("slab"), device="cpu", precision="double")
    assert np.abs(P.solve(slab)["arrays"]["T"] - P.solve(dict(slab, factorization="normal"))["arrays"]["T"]).max() < 1e-12


def test_bad_rules():
    assert not P.check(pillar(factorization="fast"))["ok"]
    assert not P.check(pillar(factorization="li"))["ok"]                                      # circular orders
    assert P.check(pillar(factorization="li", truncation="rectangular"))["ok"]


def test_wells_with_circular_orders(monkeypatch):
    stack = dataclasses.replace(tiny_stack(orders=(2, 2)), truncation="circular")
    fwd = make(stack, DEFAULT)
    rho = torch.rand(16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_(True)
    tabs = fwd.response(rho)
    t, _ = fwd.efficiencies([(rho, stack.thickness_um)])
    t = t.sum((1, 2)).reshape(2, 2) / 2 * torch.tensor([0.6, 0.4], dtype=torch.float64)
    assert torch.allclose(tabs.sum(0), t.sum(1), rtol=1e-10)
    g, = torch.autograd.grad(tabs[0].sum() - tabs[2].sum(), rho)
    d, h = g.detach() / g.norm(), 1e-5
    f = lambda r: float((lambda q: q[0].sum() - q[2].sum())(fwd.response(r)))
    assert math.isclose((f(rho.detach() + h * d) - f(rho.detach() - h * d)) / (2 * h), float(g.norm()), rel_tol=1e-6)
    full = dataclasses.replace(stack, truncation="rectangular")
    reference = make(full, DEFAULT).response(rho).detach()
    monkeypatch.setattr(Stack, "order_list", lambda self: [(m, n) for m in range(-2, 3) for n in range(-2, 3)])
    assert torch.allclose(make(stack, DEFAULT).response(rho).detach(), reference, rtol=1e-12)


def test_cuda():
    if not P.cuda_kernels_available():
        pytest.skip("needs CUDA and torch >= 2.8")
    from spectra_rcwa.kernels.gjinv import MAX_N, gj_inverse
    for n in (1, 2, 33, MAX_N):
        m = torch.randn(5, n, n, dtype=torch.complex64, device="cuda", generator=torch.Generator("cuda").manual_seed(n))
        ref = torch.linalg.inv(m.to(torch.complex128))
        assert float((gj_inverse(m) - ref).abs().max() / ref.abs().max()) < 1e-3
    stack = dataclasses.replace(STACK, truncation="circular", fourier_orders=(4, 4))
    out = []
    for kw in (dict(), dict(device="cuda", dtype=torch.complex64)):
        rho = blob().to(kw.get("device", "cpu")).to(torch.float32 if kw else torch.float64).requires_grad_(True)
        value = objective(make(stack, **kw), rho)
        out.append((float(value), torch.autograd.grad(value, rho)[0].double().cpu()))
    assert abs(out[0][0] - out[1][0]) < 1e-5 and float((out[0][1] - out[1][1]).norm() / out[0][1].norm()) < 1e-4
