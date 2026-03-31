__all__ = ["vectorize"]

from functools import wraps
from typing import Any

import mlx.core as mx

from mlxoplanet._core import asarray
from mlxoplanet.light_curves.types import LightCurveFunc
from mlxoplanet.types import Array, Scalar


def vectorize(func: LightCurveFunc) -> LightCurveFunc:
    """Vectorize a scalar light curve function over the leading time axes."""

    @wraps(func)
    def wrapped(time: Scalar, *args: Any, **kwargs: Any) -> Array | Scalar:
        time_arr = asarray(time)

        def inner(time_scalar):
            return func(time_scalar, *args, **kwargs)

        mapped = inner
        for _ in range(time_arr.ndim):
            mapped = mx.vmap(mapped)
        return mapped(time_arr)

    return wrapped
