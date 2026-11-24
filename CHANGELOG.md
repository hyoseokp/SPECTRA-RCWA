# Changelog

## 1.4.0 (2026-11-24)

Added
* `convergence_in_orders` / `problem.convergence` take a `tolerance`: the orders grow by 1.5x until the last two changes and an error estimated
  from them are below it. The result has `converged`, `fourier_orders`, `estimated_error` and, if it stops short, the `reason` (`max_matrix_size`,
  `max_seconds`, GPU memory, samples of an array).
* Patterns `array` and `npy` take `pixels: true`: constant over rectangular pixels of any number, with exact Fourier coefficients, also in the gradient.

Changed
* `problem.solve` uses exact Fourier coefficients for lamellar layers (with Li's rule in a 1D grating) and, in a crossed grating, for rectangles and
  disks, so it no longer depends on `grid` there. Sampled edges moved by up to one sample: at 512 samples a lamellar grating of fill factor 0.3 was
  off by up to 6e-3, and these efficiencies change by that. Arrays without `pixels`, the gradient with respect to the density of the shapes, and the
  Python classes are as before.
* The change reported by `convergence` is over all efficiencies, an order left out of a truncation counting as 0, instead of over the orders of the first truncation.
* The default `grid` grows with the orders above 127.
* A rectangle or disk with orders along one axis only gets a warning.

## 1.3.1 (2026-11-24)

Fixed
* `convergence_in_orders` / `problem.convergence`: the default orders repeated or went down when an axis started at 1 or 2 orders (2, 4, 4 and
  1, 3, 2), so the last change could be 0 without convergence. They now increase at every step: the problem's orders, 1.5x and 2x.
* It solves every truncation on one grid, large enough for the largest, and returns that grid. Before, a coarse `grid` grew with the orders and
  its error mixed into the change, and more than 63 orders on the default grid raised an error.
* An `orders_list` that starts below the problem's orders no longer raises a KeyError.
* The change is described as an estimate of the truncation error, not a bound.

## 1.3.0 (2026-11-23)

Changed
* 1D gratings (one of `fourier_orders` 0) build the in-plane permittivity matrices by Li's inverse rule. TM efficiencies of a lamellar grating
  now converge like the TE ones: within 1e-4 at 20 orders, where the Laurent rule still moved by 3e-4 between 80 and 160 orders.
  Crossed gratings keep the Laurent rule by default, so their results do not change. `SolverConfig(factorization=...)` picks
  `"auto"` (default), `"li"` (Li's rules also for crossed gratings) or `"laurent"` (the 1.2.0 behaviour). The adjoint gradient handles
  E_xx != E_yy (checked against autograd and finite differences).

Fixed
* `ColorRouterForward(device=None)` picked CUDA in complex64 with torch older than 2.8, which cannot compile the kernels; it now falls back to
  the CPU there, as the problem interface already did.

## 1.2.0 (2026-11-23)

Added
* `spectra_rcwa.problem`: grating problems as JSON (lattice, orders, wavelengths, incidence, media, layers with patterns), checked before solving,
  with efficiencies of every order for p and s incidence, gradients of one order's efficiency, and convergence in the orders.
* The command `spectra-rcwa` (`info`, `example`, `check`, `solve`, `mcp`) and an MCP server with seven tools (optional dependency `mcp`).
* AGENTS.md, CLAUDE.md, llms.txt, llms-full.txt, CITATION.cff, a Claude Code plugin (`agents/claude-plugin`), package keywords and links.
* `benchmarks/bench_synthetic.py`: a benchmark of the README's size on a synthetic freeform pattern, with nothing outside this package.

Removed
* The `cr_itd_v2` adapter and the benchmark scripts that needed external colour-router code and data. Their measurements stay in `benchmarks/results/`.
  No solver code changed.

## 1.1.1

Fixed
* CPU, torch with MKL on Windows (2.6.0 and 2.10.0): after `torch.set_num_threads` with more than one thread, the LU factorization of a batch of
  matrices of 150 and more returns wrong inverses or raises. A CPU batch of matrices of 128 and more is now inverted and solved one matrix at a time
  when torch has more than one thread (`spectra_rcwa.lapack`), so a solve with (4, 4) orders and more can use several threads.

## 1.1.0

Renamed
* The project is SPECTRA-RCWA: repository and distribution `spectra-rcwa`, package `spectra_rcwa` (was `eigfree-rcwa`, `eigfree_rcwa`). The environment
  variables are `SPECTRA_RCWA_CUBLAS` and `SPECTRA_BENCH_*`. Nothing else changed.

## 1.0.0

Added
* `ColorRouterForward.efficiencies`: power per diffraction order in the substrate and in the superstrate (T, R) for any stack of `stack_field`.
* `MultiLayerForward.responses` for several stacks, and `chunk` on the default route of `MultiLayerForward`.
* `device=None` picks CUDA when there is one; `cos_sinc` is exported.
* `tests/test_general.py`: efficiencies, rectangular cells and unequal orders, 1D gratings, high-index layers, argument checks, chunks, the energy guard.

Fixed
* The gradient of a layer whose index exceeds the largest transverse wave vector (few orders, a large cell, a high index or a metal) was wrong while the
  value was right: the steps and the quadrature of the adjoint integral were sized from ω max(k) alone. A step now also keeps ωh|ε|^(1/2) at most
  twice `max_step_phase`, and the Gauss-Legendre node count follows it. Cases with |ε|^(1/2) below max(k), which include every benchmark of the README
  except the two Si rows of the thick-layer table, run as before.
* The constants of the half-spaces are built in complex128 also for a complex64 solve: next to a Rayleigh point ε − kx² − ky² cancelled in float32.
* The energy guard goes on to the next nudge when a retry is not finite.
* `SolverConfig.fused_kernels` is applied by every entry point (`MultiLayerForward.response`, `stack_field`, `layer_block`), not only by `response`.
* At exactly normal incidence with a nonzero azimuth the two incident columns are x and y polarised.
* A thickness in `stack_field` can be any real number type.
* Custom backward passes are marked once-differentiable, so a second derivative raises instead of returning a wrong value.

Checked on the arguments
* `Stack`: a design grid of at least 4 × orders + 1 samples (a coarser grid wrapped the harmonics without a sign), even `detector_samples`, pairs for
  periods, orders and shapes; `Ray`: the polar angle; Im ε < 0 and an absorbing superstrate; the complex dtype; the options of the default route in
  `SolverConfig` (an odd `adjoint_nodes` gave a wrong complex64 gradient); the CUDA device against the current one; torch older than 2.8 for the
  complex64 CUDA kernels, with a message that names the alternatives; a warning when TF32 products are on.

Packaging
* Version 1.0.0; SPDX licence field; repository URL; benchmark inputs ignored by git.
