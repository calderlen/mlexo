__all__ = ["light_curve"]

from collections.abc import Callable

import mlx.core as mx

from mlxoplanet._core import asarray
from mlxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from mlxoplanet.light_curves.utils import vectorize
from mlxoplanet.proto import LightCurveOrbit
from mlxoplanet.types import Array, Scalar


def light_curve(
    orbit: LightCurveOrbit, *u: Array, order: int = 10
) -> Callable[[Scalar], Array]:
    """Compute the light curve for arbitrary polynomial limb darkening."""

    if u:
        ld_u = mx.concatenate([mx.atleast_1d(asarray(u_)) for u_ in u], axis=0)
    else:
        ld_u = mx.array([])

    @vectorize
    def light_curve_impl(time: Scalar) -> Array:
        time = asarray(time)
        if time.ndim != 0:
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{time.shape}, but a scalar was expected"
            )

        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)

        b = mx.sqrt(x**2 + y**2) / r_star
        r = orbit.radius / r_star

        lc = _limb_dark_light_curve(ld_u, b, r, order=order)
        return mx.where(z > 0, lc, mx.zeros_like(lc))

    return light_curve_impl
