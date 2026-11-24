---
name: rcwa
description: Compute diffraction efficiencies, reflection and transmission of periodic photonic structures (diffraction gratings, metagratings, metasurface unit cells, photonic-crystal slabs, thin films, stacks of patterned layers) by rigorous coupled-wave analysis (RCWA, Fourier modal method) with the open-source SPECTRA-RCWA package, and their gradients for inverse design. Use when the user asks for grating orders, diffraction efficiency, a periodic metasurface's response, or to optimise a periodic pattern.
---

# Solve periodic structures with SPECTRA-RCWA

SPECTRA-RCWA is RCWA without eigendecomposition, in PyTorch, batched on a GPU or run on the CPU, with adjoint gradients. Lengths are in µm, wavelengths in nm, angles in degrees (in problems), time dependence exp(-iωt) (absorption: Im ε > 0). The full agent guide is https://github.com/hyoseokp/spectra-rcwa/blob/main/AGENTS.md.

## Setup

1. Check the installation with `spectra-rcwa info`. If the command is missing, install `pip install "spectra-rcwa[mcp]"` (install a CUDA build of PyTorch first for GPU runs).
2. The plugin starts the `spectra-rcwa` MCP server. Its tools are `environment`, `example_problem`, `problem_schema`, `validate_problem`, `solve_efficiencies`, `efficiency_gradient` and `convergence_in_orders`. Without the server, use the command line (`spectra-rcwa example`, `check`, `solve`) or Python (`from spectra_rcwa import problem`).

## Steps

1. Call `environment` for the version, the device and the conventions.
2. Start from the closest example (`lamellar_grating`, `pillar_array`, `slab`, `two_layer_grating`) and edit the period, orders, wavelengths, incidence, permittivities and layers. Read `problem_schema` when a field is unclear. Each layer mixes two isotropic materials; rectangular lattices only.
3. Call `validate_problem` and read every warning.
4. Call `solve_efficiencies`. For a lossless stack `checks.energy_error` should be about 1e-12 in double precision.
5. Before quoting numbers, call `convergence_in_orders` and report the change between truncations (an estimate of the truncation error, not a bound); for sharp edges also refine `grid`. Metals converge slowly.
6. For sensitivities or optimisation use `efficiency_gradient` (`fd_check` verifies it).
7. Give the user the problem JSON so the run can be repeated, and the saved NPZ path.

## Citing

If the user will publish the results, tell them to cite SPECTRA-RCWA; the reference is in the README section "Citation" and in CITATION.cff.
