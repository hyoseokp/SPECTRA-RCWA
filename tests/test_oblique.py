"""Oblique lattices: the same structure in another basis or as a supercell, its symmetry, a slab, samples against exact coefficients, gradients."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from conftest import tiny_forward, tiny_stack
from spectra_rcwa import ColorRouterForward, Ray, Stack
from spectra_rcwa import problem as P
from spectra_rcwa.geometry import truncated_orders

INC = [dict(theta_deg=23.0, phi_deg=17.0)]
DISK = dict(type="disk", radius_um=0.15)


def hexagonal(**kw):
    return dict(P.example("hexagonal_pillars"), wavelengths_nm=[300.0], incidence=INC, fourier_orders=[4, 4], rayleigh_nudge=0.0,
                device="cpu", precision="double", **kw)


def table(problem, label=lambda m, n: (m, n)):
    a = P.solve(problem)["arrays"]
    return {label(int(m), int(n)): np.concatenate((a["T"][0, 0, i], a["R"][0, 0, i])) for i, (m, n) in enumerate(a["orders"])}


def largest_difference(a, b):
    zero = np.zeros(4)
    return max(np.abs(a.get(k, zero) - b.get(k, zero)).max() for k in set(a) | set(b))


def test_orders():
    assert truncated_orders((2, 3)) == [(m, n) for m in range(-2, 3) for n in range(-3, 4)]
    assert truncated_orders((2, 0), 90.0, "circular") == [(m, 0) for m in range(-2, 3)]
    assert len(truncated_orders((1, 1), 60.0)) == len(truncated_orders((1, 1), 120.0)) == 7              # (0, 0) and the 6 around it
    assert (1, 1) in truncated_orders((1, 1), 60.0) and (1, -1) in truncated_orders((1, 1), 120.0)
    for angle in (40.0, 75.0, 90.0, 120.0):
        kept = truncated_orders((4, 6), angle, "circular")
        assert kept == sorted(kept) and kept[len(kept) // 2] == (0, 0) and {(-m, -n) for m, n in kept} == set(kept)
        assert (4, 0) in kept and (0, 6) in kept and (5, 0) not in kept and (0, 7) not in kept
    for angle in (60.0, 120.0):                                             # a circle in the reciprocal plane, all of it
        s, c = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        inside = {(m, n) for m in range(-20, 21) for n in range(-20, 21) if m * m + (n - m * c) ** 2 / s ** 2 <= 25 / s ** 2 + 1e-9}
        assert set(truncated_orders((5, 5), angle)) == inside


@pytest.mark.parametrize("eps", [4.0, 12.0, dict(n=0.5, k=2.0)])
def test_the_same_lattice_in_another_basis(eps):
    a = table(hexagonal(eps_design=eps))
    b = table(hexagonal(eps_design=eps) | dict(lattice_angle_deg=120.0), lambda m, n: (m, n + m))        # a2 - a1 instead of a2
    assert len(a) == 61 and largest_difference(a, b) < 1e-12


def test_sixfold_symmetry():
    a, b = (np.sort(np.stack(list(table(hexagonal() | dict(incidence=[dict(theta_deg=23.0, phi_deg=phi)])).values())), 0) for phi in (17.0, 77.0))
    assert np.abs(a - b).max() < 1e-12
    out = P.solve(hexagonal(eps_design=12.0) | dict(wavelengths_nm=[450.0], incidence=[dict(theta_deg=0.0)]))["arrays"]
    p, s = out["T"][0, 0, len(out["orders"]) // 2]
    assert abs(p - s) < 1e-12                                               # the rectangular truncation breaks it, by 1e-2 here


def test_hexagonal_lattice_against_its_rectangular_supercell(monkeypatch):
    """Two disks in a cell of pitch x sqrt(3) pitch, with the same plane waves; the other orders of the supercell do not couple to them."""
    hexa = table(hexagonal(eps_design=12.0))
    kept = sorted((m, 2 * n - m) for m, n in hexa)
    one = P.coefficients
    monkeypatch.setattr(Stack, "order_list", lambda self: kept)
    monkeypatch.setattr(P, "_order_list", lambda *a: kept)
    monkeypatch.setattr(P, "coefficients", lambda pattern, period, m, n, angle:
                        one(dict(pattern, center_um=[0.0, 0.0]), period, m, n, angle) * (1 + torch.exp(1j * math.pi * (m + n).double())))
    cell = table(hexagonal(eps_design=12.0) | dict(lattice_angle_deg=90.0, period_um=[0.5, 0.5 * math.sqrt(3)]))
    assert largest_difference({(m, 2 * n - m): v for (m, n), v in hexa.items()}, cell) < 1e-12


def test_slab_does_not_see_the_lattice():
    slab = dict(P.example("slab"), device="cpu", precision="double", incidence=INC)
    a, b = (P.solve(dict(slab, lattice_angle_deg=angle))["results"][0]["total"] for angle in (90.0, 67.0))
    assert max(abs(a[k][s] - b[k][s]) for k in "TR" for s in "ps") < 1e-12


@pytest.mark.parametrize("pattern, angle, tol", [
    (dict(type="disk", radius_um=0.15, center_um=[0.31, 0.12]), 70.0, 5e-4), (dict(type="rectangle", size_um=[0.22, 0.3], center_um=[0.3, 0.2]), 110.0, 1.5e-3),
    (dict(type="lamellar", fill_factor=0.5, center=0.375), 70.0, 1e-4), (dict(type="lamellar", fill_factor=0.5, center=0.375, axis="y"), 130.0, 1e-4)])
def test_exact_coefficients_are_the_limit_of_the_samples(pattern, angle, tol, monkeypatch):
    problem = hexagonal() | dict(period_um=[0.5, 0.6], lattice_angle_deg=angle, fourier_orders=[2, 2], layers=[dict(thickness_um=0.4, pattern=pattern)])
    exact = table(problem)
    monkeypatch.setattr(P, "_exact", lambda *a: False)
    assert largest_difference(exact, table(dict(problem, grid=[4 * 257, 4 * 257]))) < tol


def test_shapes_that_overlap_their_images_are_sampled():
    period, orders = [0.5, 0.5], [3, 3]
    assert P._exact(dict(type="disk", radius_um=0.25), period, orders, 60.0) and not P._exact(dict(type="disk", radius_um=0.26), period, orders, 60.0)
    assert not P._exact(dict(type="disk", radius_um=0.2), [0.5, 0.5], orders, 40.0)                      # a1 - a2 is 0.34 long
    assert P._exact(dict(type="rectangle", size_um=[0.2, 0.43]), period, orders, 60.0)
    assert not P._exact(dict(type="rectangle", size_um=[0.4, 0.5]), period, orders, 60.0)
    full = P.density(dict(type="disk", radius_um=0.3), period, (64, 64), 60.0)                           # circumradius of the triangle 0.289: no gap left
    assert full.all()
    assert math.isclose(P.density(dict(type="disk", radius_um=0.15), period, (512, 512), 60.0).mean(), math.pi * 0.15 ** 2 / (0.25 * math.sin(math.pi / 3)), rel_tol=2e-3)
    assert math.isclose(P.density(dict(type="rectangle", size_um=[0.2, 0.3]), period, (512, 512), 110.0).mean(), 0.06 / (0.25 * math.sin(math.radians(110))), rel_tol=1e-2)


@pytest.mark.parametrize("pixels", [False, True])
def test_gradient(pixels):
    rho = np.random.default_rng(0).random((6, 5) if pixels else (33, 29))
    layers = [dict(thickness_um=d, pattern=dict(type="array", density=r.tolist(), pixels=pixels)) for d, r in ((0.3, rho), (0.2, 1 - rho))]
    problem = hexagonal() | dict(period_um=[0.6, 0.7], lattice_angle_deg=65.0, fourier_orders=[3, 3], grid=[33, 29], layers=layers)
    out = P.gradient(problem, order=(0, 1), polarisation="p", fd_check=True)
    assert math.isclose(out["fd_check"]["central_difference"], out["fd_check"]["gradient_norm"], rel_tol=1e-6)


def test_convergence_to_a_tolerance():
    out = P.convergence(hexagonal() | dict(wavelengths_nm=[450.0], incidence=[dict(theta_deg=0.0)], fourier_orders=[2, 2]), tolerance=3e-2)
    assert out["converged"] and out["fourier_orders"] != [2, 2]


def test_python_classes():
    stack = dataclasses.replace(tiny_stack(), lattice_angle_deg=60.0, density_shape=(16, 16))
    fwd = ColorRouterForward(stack=stack, rays=(Ray(0.2, 0.3, 1.0),), wavelengths_nm=(540.0,), device="cpu", dtype=torch.complex128)
    rho = torch.rand(16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    t, r = fwd.efficiencies([(rho, 0.4)])
    assert fwd.order_list == truncated_orders((2, 2), 60.0) and t.shape == (1, fwd.n_orders, 2)
    assert float((t.sum(1) + r.sum(1) - 1).abs().max()) < 1e-10
    with pytest.raises(ValueError, match="wells"):
        fwd.response(rho)
    for bad in (dict(lattice_angle_deg=180.0), dict(truncation="hexagonal"), dict(fourier_orders=(2, 0), density_shape=(16, 1)), dict(density_shape=(13, 13), fourier_orders=(3, 3), lattice_angle_deg=45.0)):
        with pytest.raises(ValueError):
            dataclasses.replace(stack, **bad)
    with pytest.raises(ValueError, match="factorization"):
        ColorRouterForward(stack=stack, rays=(Ray(0.2, 0.3, 1.0),), wavelengths_nm=(540.0,), device="cpu", dtype=torch.complex128,
                           config=tiny_forward().config.replace(factorization="li"))


def test_check():
    assert P.check(hexagonal())["ok"] and P.check(hexagonal())["orders"] == 61
    assert any("shorter vectors" in w for w in P.check(hexagonal() | dict(lattice_angle_deg=40.0))["warnings"])
    for bad in (dict(lattice_angle_deg=0.0), dict(truncation="round"), dict(fourier_orders=[4, 0])):
        assert not P.check(hexagonal() | bad)["ok"]


def test_single_precision_cuda():
    if not P.cuda_kernels_available():
        pytest.skip("needs CUDA and torch >= 2.8")
    a, b = (table(hexagonal() | dict(device=d, precision=q)) for d, q in (("cpu", "double"), ("cuda", "single")))
    assert largest_difference(a, b) < 3e-5
