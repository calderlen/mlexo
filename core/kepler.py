"""Core Kepler solver implemented using MLX."""

__all__ = ["kepler"]

import math

import mlx.core as mx

from mlxoplanet.types import Array


def kepler(M: Array, ecc: Array) -> tuple[Array, Array]:
    """Solve Kepler's equation to compute the true anomaly."""

    return _kepler(M, ecc)


@mx.custom_function
def _kepler(M: Array, ecc: Array) -> tuple[Array, Array]:
    M = mx.remainder(M, 2 * math.pi)

    high = M > math.pi
    M = mx.where(high, 2 * math.pi - M, M)

    ome = 1 - ecc
    E = starter(M, ecc, ome)
    E = refine(M, ecc, ome, E)
    E = mx.where(high, 2 * math.pi - E, E)

    tan_half_f = mx.sqrt((1 + ecc) / (1 - ecc)) * mx.tan(0.5 * E)
    tan2_half_f = mx.square(tan_half_f)

    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom
    return sinf, cosf


@_kepler.jvp
def _kepler_jvp(primals, tangents):
    M, e = primals
    M_dot, e_dot = tangents
    sinf, cosf = _kepler(M, e)

    ecosf = e * cosf
    ome2 = 1 - e**2

    f_dot = M_dot * (1 + ecosf) ** 2 / ome2**1.5
    f_dot += e_dot * (2 + ecosf) * sinf / ome2
    return cosf * f_dot, -sinf * f_dot


def starter(M: Array, ecc: Array, ome: Array) -> Array:
    M2 = mx.square(M)
    alpha = 3 * math.pi / (math.pi - 6 / math.pi)
    alpha += 1.6 / (math.pi - 6 / math.pi) * (math.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = mx.square(q)
    w = mx.square(mx.power(mx.abs(r) + mx.sqrt(q2 * q + r * r), 1.0 / 3.0))
    return (2 * r * w / (mx.square(w) + w * q + q2) + M) / d


def refine(M: Array, ecc: Array, ome: Array, E: Array) -> Array:
    sE = E - mx.sin(E)
    cE = 1 - mx.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (
        f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24
    )
    return E + dE
