"""Transform helpers for MLX light curve callables."""

__all__ = ["integrate", "interpolate"]

from functools import wraps
from typing import Any

import mlx.core as mx
import numpy as np

from mlxoplanet._core import asarray
from mlxoplanet.light_curves.types import LightCurveFunc
from mlxoplanet.light_curves.utils import vectorize
from mlxoplanet.types import Array, Scalar


def integrate(
    func: LightCurveFunc,
    exposure_time: Scalar | None = None,
    order: int = 0,
    num_samples: int = 7,
) -> LightCurveFunc:
    """Transform a light curve function to apply exposure time integration."""

    if exposure_time is None:
        return func

    exposure_time = asarray(exposure_time)
    if exposure_time.ndim != 0:
        raise ValueError(
            "The exposure time passed to 'integrate' must be a scalar; "
            f"got shape {exposure_time.shape}"
        )

    num_samples = int(num_samples)
    num_samples += 1 - num_samples % 2

    stencil = np.ones(num_samples, dtype=float)
    if order == 0:
        dt = np.linspace(-0.5, 0.5, 2 * num_samples + 1)[1:-1:2]
    elif order == 1:
        dt = np.linspace(-0.5, 0.5, num_samples)
        stencil *= 2
        stencil[0] = 1
        stencil[-1] = 1
    elif order == 2:
        dt = np.linspace(-0.5, 0.5, num_samples)
        stencil[1:-1:2] = 4
        stencil[2:-1:2] = 2
    else:
        raise ValueError("The parameter 'order' must be 0, 1, or 2")

    dt = mx.array(dt) * exposure_time
    stencil = mx.array(stencil / stencil.sum())

    @wraps(func)
    @vectorize
    def wrapped(time: Scalar, *args: Any, **kwargs: Any) -> Array | Scalar:
        time = asarray(time)
        if time.ndim != 0:
            raise ValueError(
                "The time passed to 'integrate' must be a scalar; "
                f"got shape {time.shape}"
            )
        result = mx.vmap(lambda t_sample: func(t_sample, *args, **kwargs))(time + dt)
        return mx.tensordot(stencil, result, axes=([0], [0]))

    return wrapped


def _linear_interp_scalar(
    x: Scalar,
    xp: Array,
    fp: Array,
    *,
    left: Array,
    right: Array,
):
    idx = mx.sum(x > xp).astype(mx.int32) - 1
    idx = mx.clip(idx, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    alpha = (x - x0) / (x1 - x0)
    interp = y0 + alpha * (y1 - y0)
    interp = mx.where(x <= xp[0], left, interp)
    interp = mx.where(x >= xp[-1], right, interp)
    return interp


def interpolate(
    func: LightCurveFunc,
    *,
    period: Scalar,
    time_transit: Scalar,
    num_samples: int,
    duration: Scalar | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> LightCurveFunc:
    """Transform a light curve function to interpolate a precomputed grid."""

    kwargs = kwargs or {}
    period = asarray(period)
    time_transit = asarray(time_transit)
    duration = period if duration is None else asarray(duration)

    time_grid = time_transit + duration * mx.linspace(-0.5, 0.5, num_samples)
    flux_grid = func(time_grid, *args, **kwargs)

    @wraps(func)
    @vectorize
    def wrapped(time: Scalar, *call_args: Any, **call_kwargs: Any) -> Array | Scalar:
        del call_args, call_kwargs
        time = asarray(time)
        time_wrapped = (
            mx.remainder(time - time_transit + 0.5 * period, period)
            + 0.5 * period
            + time_transit
        )
        return _linear_interp_scalar(
            time_wrapped,
            time_grid,
            flux_grid,
            left=flux_grid[0],
            right=flux_grid[-1],
        )

    return wrapped
