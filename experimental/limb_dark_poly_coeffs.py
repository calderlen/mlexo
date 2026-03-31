__all__ = ["calc_poly_coeffs"]

import mlx.core as mx

from mlxoplanet.types import Array


def calc_poly_coeffs(
    mu: Array,
    intensity_profile: Array,
    poly_degree: int = 10,
) -> Array:
    """Fit a polynomial limb-darkening model to an arbitrary intensity profile."""

    mu = mx.array(mu)
    intensity_profile = mx.array(intensity_profile)
    if len(mu) != len(intensity_profile):
        raise ValueError("'mu' and 'intensity_profile' must have the same length")

    sort = mx.argsort(mu)
    mu = mu[sort]
    intensity_profile = intensity_profile[sort]

    x = 1 - mu
    X = mx.stack([x**n for n in range(1, poly_degree + 1)], axis=1)
    coeffs = mx.linalg.pinv(X) @ (intensity_profile - 1.0)
    return -coeffs
