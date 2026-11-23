"""Minimal example: response and gradient of a small single-layer colour router.

Runs on CPU in a few seconds; on a CUDA GPU the complex64 path uses the fused kernels and batched inverses.
"""
import torch

from spectra_rcwa import ColorRouterForward, Stack, shifted_disk_pupil, value_and_grad

device = "cuda" if torch.cuda.is_available() else "cpu"

stack = Stack(
    period_um=(2.0, 2.0),            # one 2 x 2 pixel cell
    thickness_um=0.8,                # patterned layer thickness
    density_shape=(64, 64),          # design grid
    fourier_orders=(4, 4),           # 81 orders -> 162 x 162 layer matrices
    eps_design=2.05 ** 2,            # silicon nitride (constant index for the example)
    eps_fill=1.0,                    # air
    eps_in=1.0,                      # air superstrate
    eps_out=1.46 ** 2,               # oxide substrate
    detector_z_um=1.25,              # detector plane below the layer
)
rays = shifted_disk_pupil(numerical_aperture=0.30, chief_ray_angle_deg=2.5, chief_ray_azimuth_deg=22.5,
                          radial_order=1, azimuth_count=4)
fwd = ColorRouterForward(stack=stack, rays=rays, wavelengths_nm=(450.0, 540.0, 630.0), device=device,
                         dtype=torch.complex64)

gen = torch.Generator().manual_seed(0)
density = torch.rand(64, 64, generator=gen)

with torch.no_grad():
    tabs = fwd.response(density.to(device))                       # [well (R, G2, G1, B), wavelength]
print("Tabs:\n", tabs.cpu())

# a simple routing objective: power of each colour in its own well
target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
tabs, value, grad = value_and_grad(fwd, density, lambda t: (t * target).sum())
print(f"objective {value:.6f}, |grad| {float(grad.norm()):.3e}")
