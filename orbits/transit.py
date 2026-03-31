import mlx.core as mx

from mlxoplanet._core import asarray
from mlxoplanet.types import Scalar


class TransitOrbit:
    """An orbit parameterized directly by transit observables."""

    period: Scalar
    speed: Scalar
    duration: Scalar
    time_transit: Scalar
    impact_param: Scalar
    radius_ratio: Scalar

    def __init__(
        self,
        *,
        period: Scalar,
        duration: Scalar | None = None,
        speed: Scalar | None = None,
        time_transit: Scalar | None = None,
        impact_param: Scalar | None = None,
        radius_ratio: Scalar | None = None,
    ):
        period = asarray(period)
        duration = asarray(duration)
        speed = asarray(speed)
        time_transit = asarray(time_transit)
        impact_param = asarray(impact_param)
        radius_ratio = asarray(radius_ratio)

        if duration is None:
            if speed is None:
                raise ValueError("Either 'speed' or 'duration' must be provided")
            self.period = period
            self.speed = speed
        else:
            self.period = period
            self.duration = duration

        self.time_transit = 0.0 if time_transit is None else time_transit
        self.impact_param = 0.0 if impact_param is None else impact_param
        self.radius_ratio = 0.0 if radius_ratio is None else radius_ratio

        x2 = mx.square(1 + self.radius_ratio) - mx.square(self.impact_param)
        if duration is None:
            self.duration = 2 * mx.sqrt(mx.maximum(mx.zeros_like(x2), x2)) / self.speed
        else:
            self.speed = 2 * mx.sqrt(mx.maximum(mx.zeros_like(x2), x2)) / self.duration

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(mx.array(self.period).shape)

    @property
    def radius(self) -> Scalar:
        return self.radius_ratio

    @property
    def central_radius(self) -> Scalar:
        return mx.ones_like(self.period)

    def relative_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        del parallax

        t = asarray(t)
        half_period = 0.5 * self.period
        ref_time = self.time_transit - half_period
        dt = mx.remainder(t - ref_time, self.period) - half_period

        x = self.speed * dt
        y = mx.ones_like(dt) * self.impact_param
        m = mx.abs(dt) < 0.5 * self.duration
        z = mx.where(m, mx.ones_like(dt), -mx.ones_like(dt))
        return x, y, z
