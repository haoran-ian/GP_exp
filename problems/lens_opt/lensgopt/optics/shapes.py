import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def compute_limit_from_other(
    v1: jnp.ndarray,
    r1: jnp.ndarray,
    v2: jnp.ndarray,
    r2: jnp.ndarray,
    epsilon: jnp.ndarray,
) -> jnp.ndarray:
    if v2 - v1 < -epsilon:
        raise ValueError(
            f"Vertex of the second sphere (v2) should be not smaller than the vertex of the first sphere (v1), but v1 = {v1:.13f} and v2 = {v2:.13f}"
        )
    if v2 <= v1:  # both surfaces have the same vertex
        # v2 < v1 if a numerical error smaller than eps occures
        v2 = v1
        if jnp.sign(r1) != jnp.sign(r2) or r1 != r2:
            # half-spheres intersect only in their vertices
            return v2
        if r1 == r2 and jnp.isinf(r1):
            # the plains coincide
            return v2
        # the half-spheres coincide, so return their centers
        return v2 + r2
    # now v1 < v2
    if jnp.isinf(r1):  # the first is a plain at v1
        if jnp.isinf(r2) or r2 > 0:
            # the first is plain, the second is a plain or a half-sphere to the right of v2
            return None
        # the second half-sphere is to the left of v2
        zc_2 = v2 + r2  # center of the second half-sphere
        if zc_2 <= v1:
            # the second half-sphere is big enough to intersect the first plain
            return v1
        # the second is not big enough to intersect the plain
        return None
    # now the first is a half-sphere
    if jnp.isinf(r2):  # the second is a plain at v2
        if r1 < 0:
            # the first is half-sphere to the left of v1
            return None
        zc_1 = v1 + r1  # center of the first half-sphere
        if zc_1 >= v2:
            # the first half-sphere is big enough to intersect the second plain
            return v2
        # the second is not big enough to intersect the plain
        return None
    # both surfaces are half-spheres and their centers are at different points
    zc_1 = v1 + r1
    zc_2 = v2 + r2
    d = zc_2 - zc_1
    if jnp.abs(d) > jnp.abs(r1) + jnp.abs(r2):
        # the centers of half-spheres are too far away
        return None
    if r1 < 0 and r2 > 0:
        # half spheres look to the different sides
        return None
    # compute z_star -- potential intersection point of spheres -- half-spheres completions
    z_star = (r1**2 - r2**2 + d**2) / d * 0.5 + zc_1
    if r1**2 < (z_star - zc_1) ** 2:
        # one sphere is inside of the other
        if jnp.abs(r1) < jnp.abs(r2):  # the first sphere is inside of the second
            assert r2 < 0  # since v1 < v2
            if zc_1 > zc_2:
                # the prolonged edges of the first half-sphere intersect the second
                return zc_1
            return None
        # the second sphere is inside of the first
        assert r1 > 0  # since v1 < v2
        if zc_1 > zc_2:
            # the prolonged edges of the first half-sphere intersect the second
            return zc_2
        return None
    # the intersection between spheres happen
    # now check if it is on the correct side of half-spheres
    if z_star > zc_1 and r1 > 0:
        return None
    if z_star < zc_1 and r1 < 0:
        return None
    if z_star < zc_2 and r2 < 0:
        return None
    if z_star > zc_2 and r2 > 0:
        return None
    return z_star


def find_z_intersection_of_half_spheres_with_edges(
    v1: jnp.ndarray,
    radius1: jnp.ndarray,
    v2: jnp.ndarray,
    radius2: jnp.ndarray,
) -> jnp.ndarray:
    return None


@dataclass(frozen=True)
class RotSymSurface(ABC):
    vertex_z: jnp.ndarray

    @abstractmethod
    def ray_surface_intersection(self, ray, surface):
        pass

    @abstractmethod
    def surface_surface_intersection_z(self, other_surface: "RotSymSurface") -> float:
        pass

    @abstractmethod
    def normal(self, x, y, z):
        pass

    @abstractmethod
    def flatten_param(self) -> jnp.array:
        pass

    @abstractmethod
    def find_opposite_vertex(self) -> jnp.ndarray:
        pass

    @classmethod
    @abstractmethod
    def get_curvature_param_count(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def get_paraxial_radius(cls, flattened_params) -> float:
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)


@dataclass(frozen=True)
class Spheric(RotSymSurface):
    r: jnp.ndarray
    eps = 1e-8
    max_r = jnp.sqrt(jnp.finfo("d").max)

    @classmethod
    def create(cls, vertex_z: float, r: jnp.ndarray):
        return cls(vertex_z=vertex_z, r=r[0])

    def ray_surface_intersection(self, ray, limiting_vertex_z: jnp.ndarray):
        """
        Compute intersection of rays with this spherical (or planar) surface.
        Returns a boolean mask and the intersection points [...,3].
        """

        def planar_case(_):
            alpha = (-ray.o[..., 2] + self.vertex_z) / ray.d[..., 2]
            sol = ray.o + alpha[..., None] * ray.d
            valid = ~jnp.isnan(alpha)
            return valid, sol

        def sphere_case(_):
            d = self.vertex_z
            center = jnp.array([0.0, 0.0, d + self.r])
            e = ray.o - center

            inner_ed = jnp.sum(e * ray.d, axis=-1)
            ne2 = jnp.sum(e * e, axis=-1)
            r2 = self.r**2
            D = inner_ed**2 - (ne2 - r2)

            valid = D > 1e-8  # rays with real intersection
            sqrtD = jnp.sqrt(jnp.where(valid, D, 1.0))  # safe dummy value for AD

            # Compute alpha for converging (r > 0) or diverging (r < 0) surfaces
            alpha_raw = jnp.where(
                self.r > 0, -inner_ed - sqrtD, -inner_ed + sqrtD
            )  # omit division by nd2 = 1

            alpha = jnp.where(valid, alpha_raw, 0.0)  # mask invalid rays
            sol = ray.o + alpha[..., None] * ray.d
            sol = jnp.where(
                valid[..., None], sol, 0.0
            )  # mask invalid intersection points

            # Check z-bound validity
            z = sol[..., 2]
            lo = jnp.minimum(d, limiting_vertex_z) - self.eps
            hi = jnp.maximum(d, limiting_vertex_z) + self.eps
            valid &= (z >= lo) & (z <= hi)  # combine with existing mask

            sol = jnp.where(valid[..., None], sol, 0.0)  # reapply z-bound masking
            return valid, sol

        is_plane = jnp.isinf(self.r)
        valid, sol = jax.lax.cond(is_plane, planar_case, sphere_case, operand=None)

        return valid, sol

    def normal(self, x0, y0, z0):
        # Contract -- we guarantee that (x,y,z) belongs to the sphere to speed up the computations
        # Kidger, Eq. (3.30)
        return jnp.stack(
            [
                -x0 / self.r,
                -y0 / self.r,
                1 + (self.vertex_z - z0) / self.r,
            ],
            axis=-1,
        )

    @classmethod
    def get_curvature_param_count(cls) -> int:
        return 1

    @classmethod
    def get_paraxial_radius(cls, flattened_params: jnp.ndarray) -> float:
        return flattened_params[0]

    def surface_surface_intersection_z(self, other_surface: RotSymSurface) -> float:
        if other_surface is None:
            return jnp.nan
        other_vertex_z = other_surface.vertex_z
        if not isinstance(other_surface, Spheric):
            other_r = other_surface.__class__.get_paraxial_radius(
                other_surface.flatten_param()
            )
        else:
            other_r = other_surface.r
        return find_z_intersection_of_half_spheres_with_edges(
            self.vertex_z,
            self.r,
            other_vertex_z,
            other_r,
        )

    def flatten_param(self) -> jnp.array:
        return jnp.array([self.r])

    def find_opposite_vertex(self) -> jnp.ndarray:
        return jnp.where(jnp.isinf(self.r), self.vertex_z, self.vertex_z + self.r)


def get_surface_class_by_name(class_name: str):
    module = sys.modules[__name__]
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"No such class {class_name!r}")
    return cls


def get_curvature_param_count_via_interface(class_name: str) -> int:
    cls = get_surface_class_by_name(class_name)
    return cls.get_curvature_param_count()


def create_surface_by_name(
    class_name: str, vertice_z: jnp.ndarray, params: jnp.ndarray
) -> RotSymSurface:
    cls = get_surface_class_by_name(class_name)
    return cls.create(vertice_z, params)


def flatten_surfaces(surfaces: tuple[RotSymSurface, ...]):
    num_parameters_per_surface = []
    surface_types = []
    curvature_parameters = []
    distances_z = []
    prv_z = 0.0
    for s in surfaces:
        surface_types.append(s.__class__.__name__)
        params = s.flatten_param()
        num_parameters_per_surface.append(len(params))
        curvature_parameters.append(params)
        distances_z.append(s.vertex_z - prv_z)
        prv_z = s.vertex_z
    distances_z.pop(0)
    return (
        tuple(num_parameters_per_surface),
        tuple(surface_types),
        jnp.concat(curvature_parameters, axis=0),
        jnp.array(distances_z),
    )
