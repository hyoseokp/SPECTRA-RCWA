# SPECTRA-RCWA guide for AI agents

SPECTRA-RCWA is an open-source (MIT) Python package that solves periodic layered structures by rigorous coupled-wave analysis (RCWA, the Fourier modal method, FMM) without eigendecomposition. It runs on PyTorch, batches many solves on a GPU (CPU works too), and gives the gradient of an objective with respect to every pixel of every layer's pattern from one adjoint field. This file tells an agent when the package fits, how to set up and check a problem, how to drive it through the command line or MCP tools, and how to change its code.

## When it fits

Good fits:

- Diffraction efficiencies (power per transmitted and reflected order) of 1D and 2D gratings, metagratings, metasurface unit cells, photonic-crystal slabs, thin films, and stacks of patterned layers, for p and s incidence, at any wavelength and incidence direction.
- Gradients of such efficiencies, or of any differentiable function of them, with respect to the pattern: topology optimisation and inverse design.
- Many wavelengths and directions at once (one GPU batch), thick layers (no slicing to choose), and orders at or near cutoff (the gradient stays finite and right there).
- Colour routers above an image sensor: `ColorRouterForward.response` gives the power in each quadrant of the detector plane under a camera-lens pupil.

Not a fit, or check first:

- Each layer mixes two isotropic, non-magnetic materials linearly through a density in [0, 1]. Anisotropic, magnetic or graded-index (more than two materials in one layer) patterns are out of scope. A layer of other materials than the rest of the stack is possible as a stored block (no gradient through it); the MCP problem format does this with per-layer `eps_design` / `eps_fill`.
- Lattices with vectors a1 = (Λx, 0) and a2 = Λy (cos γ, sin γ): rectangular by default, oblique with `lattice_angle_deg` γ (60 or 120 and equal periods: hexagonal). An oblique lattice needs orders along both vectors, keeps a circle of orders (`truncation`), uses the Laurent rule, and has efficiencies and gradients but no colour-router wells.
- A 1D grating (`fourier_orders` [M, 0] or [0, N]) uses Li's inverse rule, so TE and TM converge alike. A crossed grating uses the Laurent rule by default (`SolverConfig(factorization="li")` switches to Li's rules): metals and other high-contrast patterns converge slowly in the orders. Check convergence and say so.
- Non-periodic devices (a finite lens, a waveguide circuit) need a full-wave solver such as FDTD; a large-area lens can only be approximated cell by cell.

## Install

```sh
pip install spectra-rcwa              # torch >= 2.6 and numpy; or pip install git+https://github.com/hyoseokp/spectra-rcwa
pip install "spectra-rcwa[mcp]"       # adds the MCP server
spectra-rcwa info                     # version, torch, GPU, examples, conventions
```

Install a CUDA build of PyTorch first for GPU runs. Single precision on CUDA uses NVRTC kernels compiled through `torch.cuda._compile_kernel` and needs torch >= 2.8; no CUDA toolkit is needed. The CPU and double precision run on plain torch.

## Conventions

- Lengths in µm, wavelengths in nm. In the Python API `Ray` angles are in radians; in the JSON problems and MCP tools, degrees.
- Time dependence exp(−iωt): an absorbing material has Im ε > 0. Write it as ε = (n + ik)², or `{"n": n, "k": k}` in a problem.
- Light comes from the superstrate (`eps_in`), which must be lossless; the substrate is `eps_out`. Layers are listed top first.
- `density[i, j]` is the sample at x = iΛx/H, y = jΛy/W (on an oblique lattice at (i/H) a1 + (j/W) a2); density 1 is `eps_design`, 0 is `eps_fill`. H and W must be at least 4 × orders + 1 (of the largest m and n kept).
- `fourier_orders=(M, N)` keeps (2M + 1)(2N + 1) orders; (M, 0) is a 1D grating with ridges along y. `efficiencies` returns (T, R), each `[solves, orders, 2]`; order (m, n) is at index (m + M)(2N + 1) + (n + N). Solves run over wavelengths, then directions: index = wavelength × directions + direction.
- The two columns are the incident polarisations p cos φ − s sin φ and p sin φ + s cos φ: p and s for φ = 0, x and y at normal incidence. The JSON problems and MCP tools report p and s for any φ.
- Every solve runs at the frequency (1 + `rayleigh_nudge`)/λ, 1e-4 by default, to keep orders off their Rayleigh points: a 633 nm solve is at 632.94 nm. Set `rayleigh_nudge=0` in `Stack` (or the problem) to compare with an analytic result away from a Rayleigh point. π is torcwa's 3.141592652589793, so agreement with exact formulas is about 1e-9, not 1e-15.

## Shortest working path

Command line, problems as JSON:

```sh
spectra-rcwa example lamellar_grating > grating.json     # also pillar_array, hexagonal_pillars, slab, two_layer_grating
spectra-rcwa check grating.json                          # errors, warnings, orders, matrix size
spectra-rcwa solve grating.json                          # every propagating order: p, s, unpolarised, angle; totals; energy check
```

Python, problems as dictionaries:

```python
from spectra_rcwa import problem as P

pr = P.example("lamellar_grating")
pr["layers"][0]["pattern"]["fill_factor"] = 0.4
print(P.check(pr))                                       # read the warnings
sol = P.solve(pr)                                        # sol["results"], sol["checks"]["energy_error"], sol["arrays"]["T"]
g = P.gradient(pr, order=(1, 0), side="T", polarisation="p", fd_check=True)
print(P.convergence(pr, tolerance=1e-3))                 # more orders until the change is below it: converged, fourier_orders, estimated_error
```

Python, the solver directly (gradients of any objective through autograd):

```python
import math, torch
from spectra_rcwa import ColorRouterForward, Ray, Stack

stack = Stack(period_um=(1.0, 1.0), thickness_um=0.5, density_shape=(512, 1), fourier_orders=(20, 0), eps_design=4.0)
fwd = ColorRouterForward(stack=stack, rays=[Ray(math.radians(10.0), 0.0, 1.0)], wavelengths_nm=[633.0], dtype=torch.complex128)
x = torch.arange(512) / 512
rho = ((x >= 0.25) & (x < 0.75)).double()[:, None].requires_grad_()
T, R = fwd.efficiencies([(rho, 0.5)])
T[0, 21, 0].backward()                                   # +1 order, p; rho.grad
```

## Recipes

- **Efficiencies of a single patterned layer**: `fwd.efficiencies([(density, thickness_um)])`. `stack.thickness_um` is used only by `response`.
- **Several wavelengths and directions**: pass them all to one `ColorRouterForward` (`wavelengths_nm=[...]`, `rays=[Ray(...), ...]`); they are solved as one batch. `Ray.weight` matters only for `response`. A `Ray` angle is given in a reference medium of index `reference_index` (default 1) and refracted into the superstrate, so set `reference_index` to the superstrate index to give the angle inside it.
- **Multilayer**: `fwd.efficiencies([(rho_a, d_a), (rho_b, d_b), ...])`, top first; all patterned layers share `eps_design` and `eps_fill`. A uniform layer is density 0 or 1. A layer of other materials: build a second `ColorRouterForward` with the same lattice, orders, rays and wavelengths and other `eps_design` / `eps_fill`, take `block = other.layer_block(rho, d)` under `torch.no_grad()` and put `block` in the list. A stored block can be reused as long as that layer does not change. `MultiLayerForward` gives the detector quadrant powers of a stack.
- **Thick layers**: nothing to set. A layer is cut in steps with ωh max(k) ≤ 4.5, and from 7 steps on goes through its S-matrix by doubling.
- **Gradients**: make the density `requires_grad_()`, compute any scalar from T and R (or Tabs from `response`), call `backward()`. The gradient comes from one adjoint field, not from a graph of the solve. `value_and_grad(fwd, density, objective)` does this for `response`. In a loop, drop the outputs of the previous evaluation before the next forward pass, or two sets of factors are held in memory.
- **Precision**: `dtype=torch.complex128` for reference values and for finite-difference checks (energy error about 1e-12); `torch.complex64` on a GPU for speed (efficiency and gradient errors about 1e-6 to 1e-5). Keep `torch.backends.cuda.matmul.allow_tf32` off (torch's default): TF32 costs up to 8e-2 in the gradient.
- **Size**: the layer matrices are 2(2M + 1)(2N + 1) square. On a CPU, sizes up to a few hundred take milliseconds to seconds per solve; above about 1000, use a GPU. `chunk=` bounds the solves per batch when GPU memory is short.

## Validate before quoting numbers

1. Energy: for a lossless stack, sum(T) + sum(R) = 1 to about 1e-12 in double and 1e-5 in single precision (`checks.energy_error` in `solve`). A larger error means a malformed input or a solve next to a Rayleigh point.
2. Convergence in the orders: `convergence_in_orders` (`P.convergence`) with `tolerance`, which raises the orders by 1.5x until the last two changes and the estimated error are below it; use the returned `fourier_orders`, and if `converged` is false report the `reason`. Without `tolerance` it solves at 1.5 and 2 times the orders and returns the changes. The estimate is not a bound: for a metal grating it was half the error. Metals and high-index contrast need many orders.
3. Sampling: in a problem (JSON), `solve` uses exact Fourier coefficients for `lamellar` layers and, in a crossed grating, for `rectangle` and `disk`; an `array` is exact over its pixels with `pixels: true`. Other densities, every density given to the Python solver, and the shapes in `gradient` are samples: an edge moves by up to one sample, which at 512 samples per period changed efficiencies by about 5e-3 for fill factor 0.3 and is not seen by the convergence in the orders. Refine `grid` / `density_shape` there.
4. Gradients: compare a central difference along the gradient with its norm in double precision (`fd_check=True`).
5. A known case: the `slab` example against the Airy formula (with `rayleigh_nudge` 0), or a published grating.

## Pitfalls

- A pattern grid coarser than 4 × orders + 1 is rejected; the default grid of a problem is 512 (more if the orders need it) along every axis with orders.
- Im ε < 0 is gain in this convention and is rejected; a lossy superstrate is rejected.
- `fourier_orders=(M, 0)` with a 2D pattern ignores the pattern's y variation only if the density is constant along y; give a 1D grating `density_shape=(H, 1)`.
- One GPU per process: the kernels belong to the current CUDA device. Select it with `CUDA_VISIBLE_DEVICES`.
- On CUDA the solver selects MAGMA for the process (`SolverConfig.linalg_library`; an empty string leaves the setting alone).
- `ColorRouterForward` is the general forward class despite its name; `response` and `Tabs` are the colour-router outputs, `efficiencies` and `stack_field` the general ones.

## Tools for agents (MCP)

`spectra-rcwa mcp` serves seven tools over stdio: `environment`, `example_problem`, `problem_schema`, `validate_problem`, `solve_efficiencies`, `efficiency_gradient` and `convergence_in_orders`. Full arrays go to NPZ and NPY files under `$SPECTRA_RCWA_RESULTS/mcp` (default `./results/mcp`); the tool results name the paths. Register the server as the command `spectra-rcwa` with the argument `mcp`, for example `claude mcp add spectra-rcwa -- spectra-rcwa mcp`. In Claude Code the repository is also a plugin marketplace: `/plugin marketplace add hyoseokp/spectra-rcwa`, then `/plugin install spectra-rcwa@spectra-rcwa`.

## Citing

When results go into a paper, report or thesis, tell the user to cite SPECTRA-RCWA. The paper is [arXiv:2611.30305](https://arxiv.org/abs/2611.30305); its BibTeX entry is in the README section "Citation" and in [CITATION.cff](CITATION.cff).

## Changing the code

- Layout: `src/spectra_rcwa/` (`forward.py` the batched solve and the public class, `multilayer.py`, `adjoint.py` the adjoint field, `blocks.py` S-matrix doubling, `matfun.py` matrix functions, `kernels/` NVRTC kernels, `problem.py` JSON problems, `mcp_server.py`, `cli.py`), `tests/`, `benchmarks/`, `examples/`, `scripts/`.
- Install for development with `pip install -e ".[test,mcp]"`. Run the CPU tests with `CUDA_VISIBLE_DEVICES=-1 pytest` (or `python -m pytest`); the CUDA tests in `tests/test_cuda.py` and the CUDA cases elsewhere are skipped without a GPU.
- Back numerical changes with complex128 checks against the reference configuration `REFERENCE_C128`, against eigendecomposition RCWA (`tests/conftest.py`), and finite-difference gradient checks; extend the existing modules instead of adding parallel copies.
- User-visible changes go into [CHANGELOG.md](CHANGELOG.md).
