"""Use beyond the colour router: diffraction efficiencies, a rectangular cell with unequal orders, a layer index above the largest
transverse wave vector, the checks on the arguments, chunks and several stacks, and the energy guard."""
import dataclasses
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import spectra_rcwa.forward as F
from conftest import rel, tiny_density, tiny_stack
from spectra_rcwa import DEFAULT, REFERENCE_C128, ColorRouterForward, Ray, SolverConfig, Stack, shifted_disk_pupil
from spectra_rcwa import lapack
from spectra_rcwa.multilayer import MultiLayerForward

RAYS = shifted_disk_pupil(numerical_aperture=0.3, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5, radial_order=1, azimuth_count=2)
WL = (450.0, 540.0, 640.0)


def _make(stack, cfg=DEFAULT, dtype=torch.complex128, cls=ColorRouterForward, rays=RAYS, wl=WL, **kw):
    return cls(stack=stack, rays=rays, wavelengths_nm=wl, device="cpu", dtype=dtype, config=cfg, **kw)


def _grad(fwd, rho):
    r = rho.to(torch.float64 if fwd.cdt == torch.complex128 else torch.float32).requires_grad_(True)
    tabs = fwd.response(r)
    value = (tabs * torch.linspace(-1.0, 1.0, tabs.numel(), dtype=torch.float64).reshape(tabs.shape)).sum()
    return torch.autograd.grad(value, r)[0].double()


def test_efficiencies_conserve_energy():
    rho = tiny_density()
    fwd = _make(tiny_stack())
    t, r = fwd.efficiencies([(rho, 0.6)])
    assert t.shape == (len(fwd.solve_spec), fwd.n_orders, 2) and r.shape == t.shape
    assert float((t.sum(1) + r.sum(1) - 1).abs().max()) < 1e-11
    g = fwd._solves(0, None)
    assert rel(t.sum(1), fwd._wells(fwd.stack_field([(rho, 0.6)]), g)[1]) < 1e-11       # the transmission the wells are scaled with
    absorbing = _make(dataclasses.replace(tiny_stack(), eps_design=(4.14 + 0.034j) ** 2))
    t, r = absorbing.efficiencies([(rho, 0.6)])
    assert float((t.sum(1) + r.sum(1)).max()) < 1 - 1e-3


def test_efficiencies_of_a_slab_against_airy():
    n1, n2, n3, d, wl = 1.0, 2.0, 1.46, 0.37, 0.54
    stack = Stack(period_um=(1.2, 1.2), thickness_um=d, density_shape=(16, 16), fourier_orders=(2, 2), eps_design=n2 ** 2, eps_fill=1.0, eps_in=n1 ** 2,
                  eps_out=n3 ** 2, detector_samples=(8, 8), rayleigh_nudge=0.0)
    fwd = _make(stack, rays=(Ray(0.0, 0.0, 1.0),), wl=(wl * 1000,))
    t, r = fwd.efficiencies([(torch.ones(16, 16, dtype=torch.float64), d)])
    r12, r23, ph = (n1 - n2) / (n1 + n2), (n2 - n3) / (n2 + n3), np.exp(2j * np.pi * n2 * d / wl)
    r_slab = abs((r12 + r23 * ph ** 2) / (1 + r12 * r23 * ph ** 2)) ** 2
    zeroth = fwd.n_orders // 2
    assert float((r[0, zeroth] - r_slab).abs().max()) < 1e-8 and float((t[0, zeroth] - (1 - r_slab)).abs().max()) < 1e-8
    assert float(t.sum()) - float(t[0, zeroth].sum()) < 1e-12 and float(r.sum()) - float(r[0, zeroth].sum()) < 1e-12


def test_rectangular_cell_and_unequal_orders():
    stack = Stack(period_um=(1.6, 1.0), thickness_um=0.5, density_shape=(20, 12), fourier_orders=(3, 2), eps_design=lambda wl: (2.0 + 0.01 / wl ** 2) ** 2,
                  eps_out=1.46 ** 2, detector_z_um=0.9, detector_samples=(8, 6))
    rho = tiny_density((20, 12))
    fwd = _make(stack)
    assert rel(fwd.response(rho), _make(stack, REFERENCE_C128).response(rho)) < 1e-9
    t, r = fwd.efficiencies([(rho, 0.5)])
    assert float((t.sum(1) + r.sum(1) - 1).abs().max()) < 1e-11
    grad = _grad(fwd, rho)
    assert rel(grad, _grad(_make(stack, DEFAULT.replace(handwritten_backward=False)), rho)) < 1e-9
    assert rel(_grad(_make(stack, dtype=torch.complex64), rho), grad) < 1e-4
    line = Stack(period_um=(1.2, 1.2), thickness_um=0.4, density_shape=(24, 1), fourier_orders=(5, 0), eps_design=4.0, eps_out=1.46 ** 2, detector_samples=(8, 8))
    wide = dataclasses.replace(line, density_shape=(24, 4))
    grating = tiny_density((24, 1))
    assert rel(_make(line).response(grating), _make(wide).response(grating.expand(24, 4).contiguous())) < 1e-11


@pytest.mark.parametrize("orders, eps, thickness", [((1, 1), 17.1 + 0.28j, 0.6), ((0, 0), 12.0, 0.6), ((2, 2), 17.1 + 0.28j, 0.6)])
@pytest.mark.parametrize("route", [DEFAULT.replace(block_from_steps=10 ** 6), DEFAULT.replace(block_from_steps=1)])
def test_adjoint_with_an_index_above_the_largest_wave_vector(orders, eps, thickness, route):
    """The fields of such a layer turn faster than its orders decay, and the steps and the quadrature of the adjoint integral have to follow that."""
    stack = dataclasses.replace(tiny_stack(orders=orders), eps_design=eps, thickness_um=thickness)
    rho = tiny_density()
    ref = _grad(_make(stack, route.replace(handwritten_backward=False)), rho)
    assert rel(_grad(_make(stack, route), rho), ref) < 1e-9
    assert rel(_grad(_make(stack, route, torch.complex64), rho), ref) < 1e-4


def test_arguments_are_checked():
    base = dict(period_um=(1.2, 1.2), thickness_um=0.6, density_shape=(16, 16), fourier_orders=(2, 2), eps_design=4.0)
    for bad in (dict(density_shape=(8, 16)), dict(fourier_orders=2), dict(period_um=2.0), dict(detector_samples=(9, 9)), dict(period_um=(1.2, -1.0))):
        with pytest.raises(ValueError):
            Stack(**{**base, **bad})
    for bad in (dict(adjoint_nodes=5), dict(max_step_phase=0.0), dict(cos_degree=0)):
        with pytest.raises(ValueError):
            SolverConfig(**bad)
    with pytest.raises(ValueError):
        Ray(-0.1, 0.0, 1.0)
    with pytest.raises(ValueError, match="gain"):
        _make(Stack(**{**base, "eps_design": (2.0 - 0.3j) ** 2}))
    with pytest.raises(ValueError, match="lossless"):
        _make(Stack(**{**base, "eps_in": (1.5 + 0.1j) ** 2}))
    with pytest.raises(ValueError):
        _make(Stack(**base), dtype=torch.float32)
    assert _make(Stack(**base), wl=540.0).response(tiny_density()).shape == (4, 1)


def test_chunks_and_several_stacks():
    rho = torch.stack((tiny_density(grey=False).flip(0) * 0.7 + 0.15, tiny_density()))
    whole = _make(tiny_stack(), cls=MultiLayerForward, thicknesses_um=(0.3, 0.4))
    parts = _make(tiny_stack(), cls=MultiLayerForward, thicknesses_um=(0.3, 0.4), chunk=2)
    assert rel(parts.response(rho), whole.response(rho)) < 1e-12
    assert rel(_grad(parts, rho), _grad(whole, rho)) < 1e-11
    both = whole.responses(torch.stack((rho, rho.flip(0))))
    assert rel(both[0], whole.response(rho)) < 1e-13 and rel(both[1], whole.response(rho.flip(0))) < 1e-13
    with pytest.raises(ValueError):
        whole.responses(rho)


def test_stored_block_with_any_number_as_thickness():
    rho = tiny_density()
    fwd = _make(tiny_stack(), cls=MultiLayerForward, thicknesses_um=(0.3, 0.4))
    with torch.no_grad():
        block = fwd.layer_block(rho, 0.4)
        ref = fwd.stack_field([(rho, 0.3), block])
        for d in (np.float64(0.3), torch.tensor(0.3, dtype=torch.float64)):
            assert rel(fwd.stack_field([(rho, d), block]), ref) < 1e-12


def test_normal_incidence_columns_are_x_and_y():
    fwd = _make(tiny_stack(), rays=(Ray(0.0, math.radians(30.0), 1.0),), wl=(540.0,))
    e_in, n = fwd._solves(0, None).e_in[0], fwd.n_orders
    columns = torch.stack((e_in[n // 2], e_in[n // 2 + n]))            # (Ex, Ey) of the zeroth order, one column per polarisation
    assert float((columns - torch.eye(2, dtype=columns.dtype)).abs().max()) < 1e-12


def test_energy_guard_goes_on_past_a_retry_that_is_not_finite(monkeypatch):
    """At wavelength = period / 2 and normal incidence the retry without a nudge sits on a Rayleigh point; the next nudge has to be tried."""
    rho = tiny_density()
    fwd = _make(tiny_stack(), rays=(Ray(0.0, 0.0, 1.0),), wl=(600.0,))
    ref = fwd.response(rho)
    run, calls = fwd._run, []

    def flagged(pairs, nudges, e_all, einv_all, strict=True):
        wells, trans = run(pairs, nudges, e_all, einv_all, strict)
        calls.append(tuple(nudges))
        if len(calls) == 1:
            trans = trans.clone()
            trans[0, 0] = 1.5                                          # the main solve fails the energy check
        return wells, trans

    monkeypatch.setattr(fwd, "_run", flagged)
    out = fwd.response(rho)
    assert fwd.guard_events and fwd.guard_events[0]["nudge_used"] == F.ENERGY_GUARD_RETRY_NUDGES[1]
    assert rel(out, ref) < 1e-2                                       # another frequency next to a Rayleigh point


def _after_set_num_threads():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(3, 160, 160, dtype=torch.complex128, generator=gen) + 160 * torch.eye(160)
    b = torch.randn(3, 160, 2, dtype=torch.complex128, generator=gen)
    rho = tiny_density((24, 24))
    fwd = _make(dataclasses.replace(tiny_stack(), density_shape=(24, 24), fourier_orders=(4, 4)))        # matrices of 162
    torch.set_num_threads(1)
    inverse, value, gradient = torch.linalg.inv(a), fwd.response(rho), _grad(fwd, rho)
    torch.set_num_threads(4)
    assert rel(lapack.inv(a), inverse) < 1e-12 and rel(lapack.inv_unchecked(a), inverse) < 1e-12
    assert rel(lapack.solve(a, b), inverse @ b) < 1e-12
    assert rel(fwd.response(rho), value) < 1e-10 and rel(_grad(fwd, rho), gradient) < 1e-8


def test_large_matrices_after_set_num_threads():
    """After set_num_threads, torch with MKL on Windows is wrong in the batched LU of matrices of 150 and more (spectra_rcwa.lapack).
    In a process of its own: set_num_threads cannot be taken back, and the other tests compare with torch.linalg."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", "import test_general; test_general._after_set_num_threads()"], check=True, env=env)
