# Changelog

## 1.2.0 (2026-11-23)

Added
* `spectra_rcwa.problem`: grating problems as JSON (lattice, orders, wavelengths, incidence, media, layers with patterns), checked before solving,
  with efficiencies of every order for p and s incidence, gradients of one order's efficiency, and convergence in the orders.
* The command `spectra-rcwa` (`info`, `example`, `check`, `solve`, `mcp`) and an MCP server with seven tools (optional dependency `mcp`).
* AGENTS.md, CLAUDE.md, llms.txt, llms-full.txt, CITATION.cff, a Claude Code plugin (`plugins/spectra-rcwa`), package keywords and links.
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
