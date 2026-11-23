"""Pupil quadrature of a camera lens seen from one pixel: a disk of numerical aperture NA shifted by the chief
ray, Gauss-Legendre in radius and uniform in azimuth.  Directions are in the reference medium (air by default);
weights are incoherent intensity weights normalised to cos(CRA)^4 (relative illumination)."""
from __future__ import annotations

import math

import numpy as np

from .forward import Ray


def shifted_disk_pupil(numerical_aperture: float = 0.30, chief_ray_angle_deg: float = 0.0,
                       chief_ray_azimuth_deg: float = 0.0, radial_order: int = 2, azimuth_count: int = 8,
                       reference_index: float = 1.0) -> tuple[Ray, ...]:
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    nodes, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    radius_max = numerical_aperture / reference_index
    chief_sine = math.sin(math.radians(chief_ray_angle_deg))
    chief_phi = math.radians(chief_ray_azimuth_deg)
    cx, cy = chief_sine * math.cos(chief_phi), chief_sine * math.sin(chief_phi)
    raw = []
    for node, weight in zip(nodes, weights):
        radius = radius_max * math.sqrt(float(node))
        area = math.pi * radius_max ** 2 * float(weight) / azimuth_count
        for k in range(azimuth_count):
            phi_p = math.pi / 4.0 + 2.0 * math.pi * k / azimuth_count
            sx, sy = cx + radius * math.cos(phi_p), cy + radius * math.sin(phi_p)
            st = math.hypot(sx, sy)
            if st >= 1.0:
                raise ValueError("shifted pupil contains an evanescent direction")
            theta = math.asin(st)
            raw.append((theta, math.atan2(sy, sx), area / math.cos(theta)))
    target = math.cos(math.radians(chief_ray_angle_deg)) ** 4
    total = sum(w for _, _, w in raw)
    return tuple(Ray(theta, phi, w * target / total, reference_index) for theta, phi, w in raw)
