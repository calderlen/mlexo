import mlx.core as mx

from mlxoplanet._core import asarray
from mlxoplanet.orbits.transit import TransitOrbit
from mlxoplanet.types import Scalar


def _compute_linear_ephemeris_single(transit_times: mx.array, indices: mx.array):
    """Compute linear ephemeris parameters for a single planet."""

    n = transit_times.shape[0]
    X = mx.stack([mx.ones((n,)), indices], axis=0)
    beta = mx.linalg.solve(X @ X.T, X @ transit_times)
    intercept, slope = beta
    ttvs = transit_times - (intercept + slope * indices)
    return intercept, slope, ttvs


def _compute_bin_edges_values_single(tt: mx.array, period: Scalar):
    """Compute piecewise-constant warp bins for a single planet."""

    midpoints = 0.5 * (tt[1:] + tt[:-1])
    bin_edges = mx.concatenate(
        [
            mx.array([tt[0] - 0.5 * period]),
            midpoints,
            mx.array([tt[-1] + 0.5 * period]),
        ]
    )
    bin_values = mx.concatenate([mx.array([tt[0]]), tt, mx.array([tt[-1]])])
    return bin_edges, bin_values


def _process_planet_dt_single(
    bin_edges: mx.array, bin_values: mx.array, t_mag: mx.array
) -> mx.array:
    inds = mx.sum(t_mag[..., None] > bin_edges, axis=-1).astype(mx.int32)
    return bin_values[inds]


class TTVOrbit(TransitOrbit):
    """Transit orbit with observed-minus-computed timing offsets."""

    transit_times: tuple[mx.array, ...]
    transit_inds: tuple[mx.array, ...]
    ttvs: tuple[mx.array, ...]
    t0: mx.array
    ttv_period: mx.array

    _bin_edges: tuple[mx.array, ...]
    _bin_values: tuple[mx.array, ...]

    def __init__(
        self,
        *,
        period: Scalar | None = None,
        duration: Scalar | None = None,
        speed: Scalar | None = None,
        time_transit: Scalar | None = None,
        impact_param: Scalar | None = None,
        radius_ratio: Scalar | None = None,
        transit_times: tuple[mx.array, ...] | None = None,
        transit_inds: tuple[mx.array, ...] | None = None,
        ttvs: tuple[Scalar, ...] | None = None,
        delta_log_period: float | None = None,
    ):
        if ttvs is not None and transit_times is not None:
            raise ValueError("Supply either ttvs or transit_times, not both.")
        if ttvs is None and transit_times is None:
            raise ValueError("You must supply either transit_times or ttvs.")

        if transit_times is not None:
            self.transit_times = tuple(mx.atleast_1d(asarray(tt)) for tt in transit_times)
            if transit_inds is None:
                self.transit_inds = tuple(
                    mx.arange(tt.shape[0]) for tt in self.transit_times
                )
            else:
                self.transit_inds = tuple(mx.array(inds) for inds in transit_inds)

            t0_list = []
            period_list = []
            ttvs_list = []
            for tt, inds in zip(self.transit_times, self.transit_inds, strict=False):
                t0_i, period_i, ttv_i = _compute_linear_ephemeris_single(tt, inds)
                t0_list.append(t0_i)
                period_list.append(period_i)
                ttvs_list.append(ttv_i)

            self.t0 = mx.array(t0_list)
            self.ttv_period = mx.array(period_list)
            self.ttvs = tuple(ttvs_list)
            if time_transit is None:
                time_transit = self.t0
            if period is None:
                if delta_log_period is not None:
                    period = mx.exp(mx.log(self.ttv_period) + delta_log_period)
                else:
                    period = self.ttv_period
        else:
            if period is None:
                raise ValueError("When supplying ttvs, period must be provided.")
            if time_transit is None:
                time_transit = 0.0

            self.ttvs = tuple(
                mx.atleast_1d(asarray(ttv) - mx.mean(asarray(ttv))) for ttv in ttvs
            )
            if transit_inds is None:
                self.transit_inds = tuple(mx.arange(ttv.shape[0]) for ttv in self.ttvs)
            else:
                self.transit_inds = tuple(mx.array(inds) for inds in transit_inds)

            self.t0 = mx.atleast_1d(asarray(time_transit))
            self.ttv_period = mx.atleast_1d(asarray(period))

            transit_times_list = []
            for ttv, inds in zip(self.ttvs, self.transit_inds, strict=False):
                index = len(transit_times_list)
                if self.t0.ndim > 0 and self.t0.shape[0] > 1:
                    t0_i = self.t0[index]
                    period_i = self.ttv_period[index]
                else:
                    t0_i = self.t0.item()
                    period_i = self.ttv_period.item()
                transit_times_list.append(t0_i + period_i * inds + ttv)
            self.transit_times = tuple(transit_times_list)

        super().__init__(
            period=period if period is not None else self.ttv_period,
            duration=duration,
            speed=speed,
            time_transit=time_transit,
            impact_param=impact_param,
            radius_ratio=radius_ratio,
        )

        bin_edges_list = []
        bin_values_list = []
        for tt, period_value in zip(self.transit_times, self.ttv_period, strict=False):
            edges, values = _compute_bin_edges_values_single(tt, period_value)
            bin_edges_list.append(edges)
            bin_values_list.append(values)

        self._bin_edges = tuple(bin_edges_list)
        self._bin_values = tuple(bin_values_list)

    def _get_model_dt(self, t: Scalar) -> mx.array:
        t_magnitude = mx.array(t)
        dt_list = []
        for edges, values in zip(self._bin_edges, self._bin_values, strict=False):
            dt_list.append(_process_planet_dt_single(edges, values, t_magnitude))
        return mx.stack(dt_list)

    def _warp_times(self, t: Scalar) -> Scalar:
        dt = self._get_model_dt(t)
        return t - (dt - self.t0)

    def relative_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        warped_t = self._warp_times(t)
        x, y, z = super().relative_position(warped_t, parallax=parallax)
        return mx.squeeze(x), mx.squeeze(y), mx.squeeze(z)

    @property
    def linear_t0(self):
        return mx.atleast_1d(self.t0)

    @property
    def linear_period(self):
        return mx.atleast_1d(self.ttv_period)


def compute_expected_transit_times(min_time, max_time, period, t0):
    """Compute expected transit times for each planet."""

    period = mx.atleast_1d(asarray(period))
    t0 = mx.atleast_1d(asarray(t0))

    transit_times_list = []
    for p, t0_val in zip(period, t0, strict=False):
        i_min = int(mx.ceil((min_time - t0_val) / p).item())
        i_max = int(mx.floor((max_time - t0_val) / p).item())
        indices = mx.arange(i_min, i_max + 1)
        times = t0_val + p * indices
        transit_times_list.append(times)

    return tuple(transit_times_list)
