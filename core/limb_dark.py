"""Limb-darkened light curve helpers implemented using MLX."""

__all__ = ["light_curve"]

import math
from collections.abc import Callable

import mlx.core as mx
import numpy as np
from scipy.special import binom, roots_legendre

from mlxoplanet.types import Array
from mlxoplanet.utils import zero_safe_sqrt


def light_curve(u: Array, b: Array, r: Array, *, order: int = 10):
    """Compute the light curve for arbitrary polynomial limb darkening."""

    u = mx.array(u)
    if u.ndim != 1:
        raise ValueError("Limb darkening coefficients must be a 1D array")

    if u.shape[0] == 0:
        g = mx.array([1.0 / math.pi])
    else:
        g = greens_basis_transform(u)
        g /= math.pi * (g[0] + g[1] / 1.5)

    s = solution_vector(len(g) - 1, order=order)(b, r)
    return mx.sum(s * g, axis=-1) - 1


def solution_vector(l_max: int, order: int = 10) -> Callable[[Array, Array], Array]:
    n_max = l_max + 1

    def scalar_impl(b: Array, r: Array) -> Array:
        b = mx.abs(b)
        r = mx.abs(r)
        area, kappa0, kappa1 = kappas(b, r)

        no_occ = mx.greater_equal(b, 1 + r)
        full_occ = mx.less_equal(1 + b, r)
        cond = mx.logical_or(no_occ, full_occ)
        b_ = mx.where(cond, mx.ones_like(b), b)

        b2 = mx.square(b_)
        r2 = mx.square(r)

        s0, s2 = s0s2(b_, r, b2, r2, area, kappa0, kappa1)
        s0 = mx.where(no_occ, mx.ones_like(s0) * math.pi, s0)
        s0 = mx.where(full_occ, mx.zeros_like(s0), s0)
        s2 = mx.where(cond, mx.zeros_like(s2), s2)

        pieces = [mx.expand_dims(s0, axis=0)]
        if l_max >= 1:
            P = p_integral(order, l_max, b_, r, b2, r2, kappa0)
            P = mx.where(cond, mx.zeros_like(P), P)
            pieces.append(-P[:1] - 2 * (kappa1 - math.pi) / 3)
            if l_max >= 2:
                pieces.append(mx.expand_dims(s2, axis=0))
            if l_max >= 3:
                pieces.append(-P[1:])
        return mx.concatenate(pieces, axis=0)

    def impl(b: Array, r: Array) -> Array:
        b_arr, r_arr = mx.broadcast_arrays(mx.array(b), mx.array(r))
        flat = mx.vmap(scalar_impl)(b_arr.reshape((-1,)), r_arr.reshape((-1,)))
        return flat.reshape(b_arr.shape + (n_max,))

    return impl


def greens_basis_transform(u: Array) -> Array:
    dtype = u.dtype
    u = mx.concatenate([mx.array([-1.0], dtype=dtype), u], axis=0)
    size = len(u)
    i = np.arange(size)
    arg = binom(i[None, :], i[:, None]) @ np.asarray(u)
    p = (-1) ** (i + 1) * arg
    g = [mx.zeros((), dtype=dtype) for _ in range(size + 2)]
    for n in range(size - 1, 1, -1):
        g[n] = mx.array(p[n], dtype=dtype) / (n + 2) + g[n + 2]
    g[1] = mx.array(p[1], dtype=dtype) + 3 * g[3]
    g[0] = mx.array(p[0], dtype=dtype) + 2 * g[2]
    return mx.stack(g[:-2])


def kappas(b: Array, r: Array) -> tuple[Array, Array, Array]:
    b2 = mx.square(b)
    factor = (r - 1) * (r + 1)
    cond = mx.logical_and(mx.greater(b, mx.abs(1 - r)), mx.less(b, 1 + r))
    b_ = mx.where(cond, b, mx.ones_like(b))
    area = mx.where(cond, kite_area(r, b_, mx.ones_like(r)), mx.zeros_like(r))
    return area, mx.arctan2(area, b2 + factor), mx.arctan2(area, b2 - factor)


def s0s2(
    b: Array,
    r: Array,
    b2: Array,
    r2: Array,
    area: Array,
    kappa0: Array,
    kappa1: Array,
) -> tuple[Array, Array]:
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    s0_lrg = math.pi * (1 - r2)
    s2_lrg = 2 * s0_lrg + 4 * math.pi * (eta2 - 0.5)

    Alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = math.pi - Alens
    s2_sml = 2 * s0_sml + 2 * (
        -(math.pi - kappa1) + 2 * eta2 * kappa0 - 0.25 * area * (1 + 5 * r2 + b2)
    )

    delta = 4 * b * r
    cond = mx.greater(onembpr2 + delta, delta)
    return mx.where(cond, s0_lrg, s0_sml), mx.where(cond, s2_lrg, s2_sml)


def p_integral(
    order: int,
    l_max: int,
    b: Array,
    r: Array,
    b2: Array,
    r2: Array,
    kappa0: Array,
) -> Array:
    factor = 4 * b * r
    k2_cond = mx.less(factor, 10 * mx.finfo(factor.dtype).eps)
    factor = mx.where(k2_cond, mx.ones_like(factor), factor)
    k2 = mx.maximum(mx.zeros_like(factor), (1 - r2 - b2 + 2 * b * r) / factor)

    roots, weights = roots_legendre(order)
    roots = mx.array(roots)
    weights = mx.array(weights)
    rng = 0.5 * kappa0
    phi = rng * roots
    c = mx.cos(phi + 0.5 * kappa0)

    s = mx.sin((phi + rng) / 2)
    s2 = mx.square(s)

    arg = []
    if l_max >= 1:
        omz2 = mx.maximum(mx.zeros_like(r2), r2 + b2 - 2 * b * r * c)
        z2 = 1 - omz2
        small_z = mx.less(z2, 10 * mx.finfo(omz2.dtype).eps)
        z2 = mx.where(small_z, mx.ones_like(z2), z2)
        z3 = z2 * zero_safe_sqrt(z2)
        cond = mx.less(omz2, 10 * mx.finfo(omz2.dtype).eps)
        omz2 = mx.where(cond, mx.ones_like(z2), omz2)
        result = 2 * r * (r - b * c) * (1 - z3) / (3 * omz2)
        arg.append(mx.where(cond, mx.zeros_like(result[None, :]), result[None, :]))
    if l_max >= 3:
        f0 = mx.maximum(
            mx.zeros_like(r2), mx.where(k2_cond, 1 - r2, factor * (k2 - s2))
        )
        n = mx.arange(3, l_max + 1)
        f = f0[None, :] ** (0.5 * n[:, None])
        f *= 2 * r * (r - b + 2 * b * s2[None, :])
        arg.append(f)

    return rng * mx.sum(mx.concatenate(arg, axis=0) * weights[None, :], axis=1)


def kite_area(a: Array, b: Array, c: Array) -> Array:
    def sort2(x: Array, y: Array) -> tuple[Array, Array]:
        return mx.minimum(x, y), mx.maximum(x, y)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return zero_safe_sqrt(mx.maximum(mx.zeros_like(square_area), square_area))
