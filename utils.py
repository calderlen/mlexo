__all__ = ["asarray", "get_dtype_eps", "zero_safe_sqrt"]

import mlx.core as mx

from mlxoplanet._core import asarray


def get_dtype_eps(x):
    return mx.finfo(asarray(x).dtype).eps


@mx.custom_function
def zero_safe_sqrt(x):
    return mx.sqrt(x)


@zero_safe_sqrt.jvp
def zero_safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = mx.sqrt(x)
    cond = mx.less(x, 10 * get_dtype_eps(x))
    denom = mx.where(cond, mx.ones_like(x), x)
    return 0.5 * x_dot * primal_out / denom
