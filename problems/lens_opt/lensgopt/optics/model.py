from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import jax.numpy as jnp

import lensgopt.materials.refractive_index_catalogs as ior_catalogs
import lensgopt.optics.meta as meta


class Aperture(NamedTuple):
    """
    Defines the aperture stop configuration for an optical system.

    The aperture controls the cone of light that can pass through the system,
    influencing illumination, depth of field, and aberrations.

    Attributes:
        type (str): Aperture type. Supported options:
            - 'ED': Entrance pupil diameter.
            - 'float': Floating aperture.
            - 'objectNA': Object numerical aperture.
            - 'fNum': F-number specification.
            - 'parFNum': Paraxial F-number.
        MAX_D (float | None): Maximum diameter of the aperture stop, in millimeters.
            If None, the aperture is considered unbounded.
    """

    type: str = "ED"  # options: 'ED', 'float', 'objectNA', 'fNum', 'parFNum'
    max_d: float = None  # [mm]


class Field(NamedTuple):
    """
    Defines the object field configuration for an optical system.

    The field specification determines how rays originate from the object
    space, which can be angular or spatial, affecting image height and field curvature.

    Attributes:
        type (str): Field type. Supported options:
            - 'angle': Field specified by chief ray angle (degrees).
            - 'objectHeight': Field specified by object height (millimeters).
        MAX_FIELD (float | None): Maximum field value. For 'angle', this is the angle in degrees;
            for 'objectHeight', this is the object height in millimeters. If None,
            the field is unbounded.
    """

    type: str = "angle"  # options: 'angle', 'objectHeight'
    max_field: float = None  # angle in degrees, height in mm


class LensSystemConstants(NamedTuple):
    """
    Holds all fixed design parameters and target specifications for an optical lens system.

    This dataclass encapsulates:
      - Definitions of the field and aperture.
      - Number and types of surfaces in the lens train.
      - Material catalogs for each surface.
      - Target (desired) values for performance metrics.
      - Wavelengths and field weighting factors.
      - Paraxial thresholds and mode for computing sensor position.

    Attributes:
        lens_field (Field):
            Defines the object field type ('angle' or 'objectHeight') and maximum extent.
        aperture (Aperture):
            Specifies the aperture stop type and its physical size (diameter).
        num_surfaces (int):
            Total number of optical surfaces in the system.
        num_parameters_per_surface (jnp.ndarray):
            Array of length `num_surfaces` indicating how many design parameters each surface has.
        surface_type_id (jnp.ndarray):
            Array of length `num_surfaces` indicating surface type codes (e.g., 0 = Spheric, 1 = Aspheric).
        catalogs (Sequence[OpticalMaterialCatalog]):
            Material catalog instance assigned to each surface (length `num_surfaces`).
        aperture_stop_index (int):
            Index of the surface that serves as the aperture stop (0-based).
        target_effective_focal_length (float):
            Desired effective focal length (mm).
        target_edge_thickness (float):
            Desired edge thickness (mm) at the sensor.
        wavelengths (jnp.ndarray):
            1D array of wavelengths (in nanometers) for which refractive indices are precomputed.
            The first element must be the standard D-line (meta.LAMBDA_D).
        field_factors (jnp.ndarray):
            1D array of weighting factors for different field points (e.g., [0.0, 0.7, 1.0]).
        MAX_PARAXIAL_ANGLE (float):
            Maximum allowed paraxial angle (radians) for marginal-ray checks.
        EFFL_PARAXIAL_Y (float):
            Y-coordinate tolerance for paraxial effective focal length computation.
        object_z (float):
            Z-coordinate of the object plane (defaults to negative infinity).
        sensor_mode (str):
            Mode for determining sensor position:
              - 'optimize' to treat sensor_z as a free variable
              - 'paraxial_solve' to solve for sensor_z via paraxial optics
    """

    # Field and aperture definitions
    lens_field: Field
    aperture: Aperture

    # Number of surfaces and parameters per surface
    num_surfaces: int
    num_parameters_per_surface: tuple  # shape = (num_surfaces,)
    surface_types: tuple[str, ...]  # shape = (num_surfaces,) (Spheric, Aspheric, ...)

    # Refractive index catalogs assigned to each surface
    ior_catalogs: tuple[ior_catalogs.RefractiveIndexCatalog, ...]
    aperture_stop_index: int

    # Target performance values
    target_effective_focal_length: float  # [mm]
    target_edge_thickness: float  # [mm]
    target_axis_thickness: float  # [mm]
    target_free_working_distance: float  # [mm]
    # ... additional target_* values can be added here ...

    # Wavelengths (first element must be the D-line)
    wavelengths: tuple = tuple(
        [
            meta.LAMBDA_D,  # Must always be first (Fraunhofer D-line)
            meta.LAMBDA_C,
            meta.LAMBDA_F,
        ]
    )

    # Weight factors for different field points
    field_factors: tuple = tuple([0.0, 0.7, 1.0])
    edge_thickness_field_factors: tuple = tuple([-1, 0, 1])
    edge_thickness_entrance_pupil_factors: tuple = tuple([-1, 1])

    # Paraxial threshold constants
    MAX_PARAXIAL_ANGLE: float = 1e-3
    EFFL_PARAXIAL_Y: float = 1e-6

    # Object and sensor z coordinates
    object_z: float = float("-inf")
    sensor_z: float = 0.0

    # Mode for computing sensor_z: 'fixed', 'optimize' or 'paraxial_solve'
    sensor_z_mode: str = "paraxial_solve"

    # Mode for computing gradients w.r.t. iors
    is_iors_grad: bool = False

    # Number of rays in one row of square pattern
    n_rays_row: int = 31

    def __post_init__(self):
        # 1) Ensure the first wavelength is meta.LAMBDA_D (D-line)
        if len(self.wavelengths) < 1 or self.wavelengths[0] != meta.LAMBDA_D:
            raise ValueError("The first element of wavelengths must be meta.LAMBDA_D")

        # 2) Validate array lengths match the declared number of surfaces
        if len(self.num_parameters_per_surface) != self.num_surfaces:
            raise ValueError("num_parameters_per_surface must have length num_surfaces")
        if len(self.surface_types) != self.num_surfaces:
            raise ValueError("surface_type_id must have length num_surfaces")
        if len(self.ior_catalogs) != self.num_surfaces + 1:
            raise ValueError("len(catalogs) must equal num_surfaces + 1")

        # 3) Additional validation for target values
        if self.target_effective_focal_length <= 0:
            raise ValueError("target_effective_focal_length must be > 0")
        if self.target_edge_thickness < 0:
            raise ValueError("target_edge_thickness must be >= 0")

        # 4) aperture_stop_index must be within valid range
        if not (0 <= self.aperture_stop_index < self.num_surfaces):
            raise ValueError("aperture_stop_index must be in [0, num_surfaces)")

        # 5) sensor_mode must be either 'optimize' or 'paraxial_solve'
        if self.sensor_z_mode not in ("fixed", "optimize", "paraxial_solve"):
            raise ValueError(
                "sensor_mode must be 'fixed', 'optimize' or 'paraxial_solve'"
            )


class LensSystemVariables(NamedTuple):
    curvature_parameters: jnp.ndarray
    distances_z: jnp.ndarray  # shape (num_surf - 1,)
    iors: jnp.ndarray  # shape (num_surf + 1, ) this is used to compute iors

    @staticmethod
    def resolve_verticies_z(distances_z):
        zero = jnp.zeros((1,))
        prefs = jnp.cumsum(distances_z)
        return jnp.concatenate([zero, prefs], axis=0)


class LensSystemComputedProperties(NamedTuple):
    # Does not include expensive metrics like rms_spot and others
    iors: jnp.ndarray
    entrance_pupil_z: float
    entrance_pupil_diameter: float
    effective_focal_length: float
    paraxial_marginal_ray: jnp.ndarray  # [y, alpha]
    vertices_z: jnp.ndarray
    limiting_vertices_z: jnp.ndarray
    up_corners_y: jnp.ndarray
    sensor_z: float
