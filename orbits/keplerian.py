"""Keplerian orbits implemented against MLX arrays."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import math

import mlx.core as mx

from mlxoplanet import constants
from mlxoplanet._core import asarray, ndim
from mlxoplanet.core.kepler import kepler
from mlxoplanet.object_stack import ObjectStack
from mlxoplanet.types import Scalar


class Central:
    """A central body in an orbital system."""

    mass: Scalar | None
    radius: Scalar | None
    density: Scalar | None

    def __init__(
        self,
        *,
        mass: Scalar | None = None,
        radius: Scalar | None = None,
        density: Scalar | None = None,
    ):
        mass = asarray(mass)
        radius = asarray(radius)
        density = asarray(density)

        if radius is None and mass is None:
            radius = mx.array(1.0)
            if density is None:
                mass = mx.array(1.0)

        if any(ndim(arg) != 0 for arg in (mass, radius, density) if arg is not None):
            raise ValueError("All parameters of a KeplerianCentral must be scalars")

        error_msg = (
            "Values must be provided for exactly two of mass, radius, and density"
        )
        if density is None:
            if mass is None or radius is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = radius
            self.density = 3 * mass / (4 * math.pi * radius**3)
        elif radius is None:
            if mass is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = (3 * mass / (4 * math.pi * density)) ** (1 / 3)
            self.density = density
        elif mass is None:
            self.mass = 4 * math.pi * radius**3 * density / 3.0
            self.radius = radius
            self.density = density
        else:
            raise ValueError(error_msg)

    @classmethod
    def from_orbital_properties(
        cls,
        *,
        period: Scalar,
        semimajor: Scalar,
        radius: Scalar | None = None,
        body_mass: Scalar | None = None,
    ) -> "Central":
        period = asarray(period)
        semimajor = asarray(semimajor)
        radius = asarray(radius)
        body_mass = asarray(body_mass)

        if any(
            ndim(arg) != 0 for arg in (semimajor, period, body_mass) if arg is not None
        ):
            raise ValueError(
                "All parameters of 'Central.from_orbital_properties' must be scalars"
            )

        radius = mx.array(1.0) if radius is None else radius
        mass = 4 * math.pi**2 * semimajor**3 / (constants.G * period**2)
        if body_mass is not None:
            mass -= body_mass
        return cls(mass=mass, radius=radius)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(mx.array(self.mass).shape)


class Body:
    """Initialize an orbiting body using orbital parameters."""

    time_transit: Scalar | None
    time_peri: Scalar | None
    period: Scalar | None
    semimajor: Scalar | None
    inclination: Scalar | None
    impact_param: Scalar | None
    eccentricity: Scalar | None
    omega_peri: Scalar | None
    sin_omega_peri: Scalar | None
    cos_omega_peri: Scalar | None
    asc_node: Scalar | None
    sin_asc_node: Scalar | None
    cos_asc_node: Scalar | None
    mass: Scalar | None
    radius: Scalar | None
    radial_velocity_semiamplitude: Scalar | None
    parallax: Scalar | None

    def __init__(
        self,
        *,
        time_transit: Scalar | None = None,
        time_peri: Scalar | None = None,
        period: Scalar | None = None,
        semimajor: Scalar | None = None,
        inclination: Scalar | None = None,
        impact_param: Scalar | None = None,
        eccentricity: Scalar | None = None,
        omega_peri: Scalar | None = None,
        sin_omega_peri: Scalar | None = None,
        cos_omega_peri: Scalar | None = None,
        asc_node: Scalar | None = None,
        sin_asc_node: Scalar | None = None,
        cos_asc_node: Scalar | None = None,
        mass: Scalar | None = None,
        radius: Scalar | None = None,
        radial_velocity_semiamplitude: Scalar | None = None,
        parallax: Scalar | None = None,
    ):
        self.time_transit = asarray(time_transit)
        self.time_peri = asarray(time_peri)
        self.period = asarray(period)
        self.semimajor = asarray(semimajor)
        self.inclination = asarray(inclination)
        self.impact_param = asarray(impact_param)
        self.eccentricity = asarray(eccentricity)
        self.omega_peri = asarray(omega_peri)
        self.sin_omega_peri = asarray(sin_omega_peri)
        self.cos_omega_peri = asarray(cos_omega_peri)
        self.asc_node = asarray(asc_node)
        self.sin_asc_node = asarray(sin_asc_node)
        self.cos_asc_node = asarray(cos_asc_node)
        self.mass = asarray(mass)
        self.radius = asarray(radius)
        self.radial_velocity_semiamplitude = asarray(radial_velocity_semiamplitude)
        self.parallax = asarray(parallax)
        self.__check_init__()

    def __check_init__(self) -> None:
        if not ((self.period is None) ^ (self.semimajor is None)):
            raise ValueError("Exactly one of period or semimajor must be specified")

        provided_input_arguments = [
            arg
            for arg in (
                self.time_transit,
                self.time_peri,
                self.period,
                self.semimajor,
                self.inclination,
                self.impact_param,
                self.eccentricity,
                self.omega_peri,
                self.sin_omega_peri,
                self.cos_omega_peri,
                self.asc_node,
                self.sin_asc_node,
                self.cos_asc_node,
                self.mass,
                self.radius,
                self.radial_velocity_semiamplitude,
                self.parallax,
            )
            if arg is not None
        ]
        if any(ndim(arg) != 0 for arg in provided_input_arguments):
            raise ValueError(
                "All input arguments to 'Body' must be scalars; for multi-body systems "
                "use 'System'"
            )

        if self.omega_peri is not None and (
            self.sin_omega_peri is not None or self.cos_omega_peri is not None
        ):
            raise ValueError(
                "Cannot specify both omega_peri and sin_omega_peri or cos_omega_peri"
            )
        if (self.sin_omega_peri is not None) ^ (self.cos_omega_peri is not None):
            raise ValueError("Both sin_omega_peri and cos_omega_peri must be specified")

        if self.asc_node is not None and (
            self.sin_asc_node is not None or self.cos_asc_node is not None
        ):
            raise ValueError(
                "Cannot specify both asc_node and sin_asc_node or cos_asc_node"
            )
        if (self.sin_asc_node is not None) ^ (self.cos_asc_node is not None):
            raise ValueError("Both sin_asc_node and cos_asc_node must be specified")

        has_omega_peri = (
            self.omega_peri is not None
            or self.sin_omega_peri is not None
            or self.cos_omega_peri is not None
        )
        if (self.eccentricity is not None) ^ has_omega_peri:
            raise ValueError(
                "Both or neither of eccentricity and omega_peri must be specified"
            )

        if self.impact_param is not None and self.inclination is not None:
            raise ValueError(
                "Only one of impact_param and inclination can be specified"
            )

        if self.time_transit is not None and self.time_peri is not None:
            raise ValueError("Only one of time_transit or time_peri can be specified")


class OrbitalBody:
    """A computational representation of an orbiting body."""

    central: Central
    time_ref: Scalar
    time_transit: Scalar
    period: Scalar
    semimajor: Scalar
    sin_inclination: Scalar
    cos_inclination: Scalar
    impact_param: Scalar
    mass: Scalar | None
    radius: Scalar | None
    eccentricity: Scalar | None
    sin_omega_peri: Scalar | None
    cos_omega_peri: Scalar | None
    sin_asc_node: Scalar | None
    cos_asc_node: Scalar | None
    radial_velocity_semiamplitude: Scalar | None
    parallax: Scalar | None

    def __init__(self, central: Central, body: Body):
        self.central = central
        self.radius = body.radius
        self.mass = body.mass
        self.radial_velocity_semiamplitude = body.radial_velocity_semiamplitude
        self.parallax = body.parallax

        mass_factor = constants.G * self.total_mass
        if body.semimajor is None:
            assert body.period is not None
            self.semimajor = (mass_factor * body.period**2 / (4 * math.pi**2)) ** (
                1 / 3
            )
            self.period = body.period
        elif body.period is None:
            self.semimajor = body.semimajor
            self.period = (
                2 * math.pi * body.semimajor * mx.sqrt(body.semimajor / mass_factor)
            )
        else:
            raise ValueError("Exactly one of period or semimajor must be specified")

        if body.omega_peri is not None:
            self.sin_omega_peri = mx.sin(body.omega_peri)
            self.cos_omega_peri = mx.cos(body.omega_peri)
        else:
            self.sin_omega_peri = body.sin_omega_peri
            self.cos_omega_peri = body.cos_omega_peri

        if body.asc_node is not None:
            self.sin_asc_node = mx.sin(body.asc_node)
            self.cos_asc_node = mx.cos(body.asc_node)
        else:
            self.sin_asc_node = body.sin_asc_node
            self.cos_asc_node = body.cos_asc_node

        self.eccentricity = body.eccentricity
        if self.eccentricity is None:
            M0 = mx.ones_like(self.period) * (0.5 * math.pi)
            incl_factor = 1.0
        else:
            assert self.sin_omega_peri is not None
            assert self.cos_omega_peri is not None
            opsw = 1 + self.sin_omega_peri
            E0 = 2 * mx.arctan2(
                mx.sqrt(1 - self.eccentricity) * self.cos_omega_peri,
                mx.sqrt(1 + self.eccentricity) * opsw,
            )
            M0 = E0 - self.eccentricity * mx.sin(E0)

            ome2 = 1 - self.eccentricity**2
            incl_factor = (1 + self.eccentricity * self.sin_omega_peri) / ome2

        dcosidb = incl_factor * central.radius / self.semimajor
        if body.impact_param is not None:
            self.impact_param = body.impact_param
            self.cos_inclination = dcosidb * body.impact_param
            self.sin_inclination = mx.sqrt(1 - self.cos_inclination**2)
        elif body.inclination is not None:
            self.cos_inclination = mx.cos(body.inclination)
            self.sin_inclination = mx.sin(body.inclination)
            self.impact_param = self.cos_inclination / dcosidb
        else:
            z = mx.zeros_like(self.period)
            self.impact_param = z
            self.cos_inclination = z
            self.sin_inclination = mx.ones_like(self.period)

        self.time_ref = -M0 * self.period / (2 * math.pi)
        if body.time_transit is not None:
            self.time_transit = body.time_transit
        elif body.time_peri is not None:
            self.time_transit = body.time_peri - self.time_ref
        else:
            self.time_transit = mx.zeros_like(self.time_ref)

    @property
    def central_radius(self) -> Scalar:
        return self.central.radius

    @property
    def time_peri(self) -> Scalar:
        return self.time_transit + self.time_ref

    @property
    def inclination(self) -> Scalar:
        return mx.arctan2(self.sin_inclination, self.cos_inclination)

    @property
    def omega_peri(self) -> Scalar | None:
        if self.eccentricity is None:
            return None
        return mx.arctan2(self.sin_omega_peri, self.cos_omega_peri)

    @property
    def total_mass(self) -> Scalar:
        return self.central.mass if self.mass is None else self.mass + self.central.mass

    def position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        semimajor = -self.semimajor * self.central.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def central_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        semimajor = self.semimajor * self.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def relative_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        return self._get_position_and_velocity(
            t,
            semimajor=-self.semimajor,
            parallax=parallax,
        )[0]

    def relative_angles(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar]:
        X, Y, _ = self.relative_position(t, parallax=parallax)
        rho = mx.sqrt(X**2 + Y**2)
        theta = mx.arctan2(Y, X)
        return rho, theta

    def velocity(
        self, t: Scalar, semiamplitude: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        if semiamplitude is None:
            mass: Scalar = -self.central.mass
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.central.mass / self.mass
        return self._get_position_and_velocity(t, semiamplitude=k)[1]

    def central_velocity(
        self, t: Scalar, semiamplitude: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        if semiamplitude is None:
            return self._get_position_and_velocity(t, mass=self.mass)[1]
        return self._get_position_and_velocity(t, semiamplitude=semiamplitude)[1]

    def relative_velocity(
        self, t: Scalar, semiamplitude: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        if semiamplitude is None:
            mass: Scalar = -self.total_mass
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.total_mass / self.mass
        return self._get_position_and_velocity(t, semiamplitude=k)[1]

    def radial_velocity(self, t: Scalar, semiamplitude: Scalar | None = None) -> Scalar:
        return -self.central_velocity(t, semiamplitude=semiamplitude)[2]

    def _warp_times(self, t: Scalar) -> Scalar:
        return t - self.time_transit

    def _get_true_anomaly(self, t: Scalar) -> tuple[Scalar, Scalar]:
        M = 2 * math.pi * (self._warp_times(t) - self.time_ref) / self.period
        if self.eccentricity is None:
            return mx.sin(M), mx.cos(M)
        return kepler(M, self.eccentricity)

    def _rotate_vector(
        self, x: Scalar, y: Scalar, *, include_inclination: bool = True
    ) -> tuple[Scalar, Scalar, Scalar]:
        if self.eccentricity is None:
            x1 = x
            y1 = y
        else:
            x1 = self.cos_omega_peri * x - self.sin_omega_peri * y
            y1 = self.sin_omega_peri * x + self.cos_omega_peri * y

        if include_inclination:
            x2 = x1
            y2 = self.cos_inclination * y1
            Z = -self.sin_inclination * y1
        else:
            x2 = x1
            y2 = y1
            Z = -y1

        if self.cos_asc_node is None:
            return x2, y2, Z

        X = self.cos_asc_node * x2 - self.sin_asc_node * y2
        Y = self.sin_asc_node * x2 + self.cos_asc_node * y2
        return X, Y, Z

    def _get_position_and_velocity(
        self,
        t: Scalar,
        semimajor: Scalar | None = None,
        mass: Scalar | None = None,
        semiamplitude: Scalar | None = None,
        parallax: Scalar | None = None,
    ) -> tuple[tuple[Scalar, Scalar, Scalar], tuple[Scalar, Scalar, Scalar]]:
        if semiamplitude is None:
            semiamplitude = self.radial_velocity_semiamplitude

        if parallax is None:
            parallax = self.parallax

        if semiamplitude is None:
            if self.radial_velocity_semiamplitude is None:
                m = 1.0 if mass is None else mass
                k0 = 2 * math.pi * self.semimajor * m / (self.total_mass * self.period)
                if self.eccentricity is not None:
                    k0 /= mx.sqrt(1 - self.eccentricity**2)
            else:
                k0 = self.radial_velocity_semiamplitude

            if parallax is not None:
                k0 = k0 * constants.au * parallax
        else:
            k0 = semiamplitude

        r0 = 1.0
        if semimajor is not None:
            if parallax is None:
                r0 = semimajor
            else:
                r0 = semimajor * constants.au * parallax

        sinf, cosf = self._get_true_anomaly(t)
        if self.eccentricity is None:
            v1, v2 = -k0 * sinf, k0 * cosf
        else:
            v1, v2 = -k0 * sinf, k0 * (cosf + self.eccentricity)
            r0 *= (1 - self.eccentricity**2) / (1 + self.eccentricity * cosf)

        x, y, z = self._rotate_vector(r0 * cosf, r0 * sinf)
        vx, vy, vz = self._rotate_vector(
            v1, v2, include_inclination=semiamplitude is None
        )
        return (x, y, z), (vx, vy, vz)


class System:
    """A Keplerian orbital system."""

    central: Central
    _body_stack: ObjectStack[OrbitalBody]

    def __init__(
        self,
        central: Central | None = None,
        *,
        bodies: Iterable[Body | OrbitalBody] = (),
    ):
        self.central = Central() if central is None else central
        self._body_stack = ObjectStack(
            *(
                body if isinstance(body, OrbitalBody) else OrbitalBody(self.central, body)
                for body in bodies
            )
        )

    def __repr__(self) -> str:
        return f"System(central={self.central!r}, bodies={self.bodies!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._body_stack),)

    @property
    def bodies(self) -> tuple[OrbitalBody, ...]:
        return self._body_stack.objects

    @property
    def radius(self) -> Scalar:
        return self.body_vmap(lambda body: body.radius, in_axes=None)()

    @property
    def central_radius(self) -> Scalar:
        return self.body_vmap(lambda body: body.central_radius, in_axes=None)()

    def add_body(
        self,
        body: Body | None = None,
        central: Central | None = None,
        **kwargs: Any,
    ) -> "System":
        body_ = body
        if body_ is None:
            body_ = Body(**kwargs)
        if central is not None:
            body_ = OrbitalBody(central, body_)
        return System(central=self.central, bodies=self.bodies + (body_,))

    def body_vmap(
        self,
        func: Callable,
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._body_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)

    def position(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.position, in_axes=None)(t)

    def central_position(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.central_position, in_axes=None)(t)

    def relative_position(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.relative_position, in_axes=None)(t)

    def velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.velocity, in_axes=None)(t)

    def central_velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.central_velocity, in_axes=None)(t)

    def relative_velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
        return self.body_vmap(OrbitalBody.relative_velocity, in_axes=None)(t)

    def radial_velocity(self, t: Scalar) -> Scalar:
        return self.body_vmap(OrbitalBody.radial_velocity, in_axes=None)(t)
