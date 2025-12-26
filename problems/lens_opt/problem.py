# fmt: off
from importlib import reload

import os
import sys
import ioh
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lensgopt.materials.refractive_index_catalogs as refractive_index_catalogs
import lensgopt.optics.loss as loss
import lensgopt.optics.meta as meta
import lensgopt.optics.model as model
import lensgopt.optics.optics as optics
import lensgopt.optics.shapes as shapes
import lensgopt.parsers.lens_creation as lens_creation
import lensgopt.parsers.zmax_fmt as zmax_fmt

reload(zmax_fmt)
reload(refractive_index_catalogs)
reload(meta)
reload(model)
reload(optics)
reload(shapes)
reload(loss)
reload(lens_creation)
# fmt: on


class lens_opt:

    def __init__(self):
        cnst, vars, material_ids = lens_creation.create_double_gauss(
            sensor_z_mode="paraxial_solve"
        )
        len_catalog = 120
        curvatures_optimized_idx = jnp.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
        distances_optimized_idx = jnp.array([0, 1, 2, 3, 6, 7, 8, 9])
        # materials_optimized_idx = jnp.array([1, 3, 4, 7, 8, 10])
        self.materials_optimized_idx = jnp.empty(0)

        low_distances = []
        high_distances = []
        internal_idx = set([0, 2, 3, 6, 7, 9])
        for i in np.asarray(distances_optimized_idx):
            if i in internal_idx:
                low_distance = 0.5
                high_distance = 15.0
            else:
                low_distance = 0.5
                high_distance = 20.0
            low_distances.append(low_distance)
            high_distances.append(high_distance)

        self.original_lb = (
            [-0.1] * curvatures_optimized_idx.size
            + low_distances
            + [0] * self.materials_optimized_idx.size
        )
        self.original_ub = (
            [0.1] * curvatures_optimized_idx.size
            + high_distances
            + [len_catalog - 1] * self.materials_optimized_idx.size
        )
        self.dim = len(self.original_lb)

        self.lb = (
            [-1.0] * curvatures_optimized_idx.size
            + [-1.0] * distances_optimized_idx.size
            + [0] * self.materials_optimized_idx.size
        )
        self.ub = (
            [1.0] * curvatures_optimized_idx.size
            + [1.0] * distances_optimized_idx.size
            + [len_catalog - 1] * self.materials_optimized_idx.size
        )

        forward_transform, backward_transform = loss.create_affine_transforms(
            (jnp.array(self.original_lb), jnp.array(self.original_ub)),
            (jnp.array(self.lb), jnp.array(self.ub)),
        )

        self.resolver = loss.InverseCurvatureParameterBlockResolver(
            curvatures_template=vars.curvature_parameters,
            distances_template=vars.distances_z,
            iors_template=vars.iors,
            material_ids_template=material_ids,
            curvatures_optimized_idx=curvatures_optimized_idx,
            distances_optimized_idx=distances_optimized_idx,
            materials_optimized_idx=self.materials_optimized_idx,
            catalogs=cnst.ior_catalogs,
            transform=forward_transform,
            inverse_transform=backward_transform,
            is_inverse_curvatures=True,
        )
        self.factory = loss.AutoLossFactory(
            loss_full=lambda flat_params_, dists_, iors_: loss.loss(
                cnst, flat_params_, dists_, iors_
            ),
            resolver=self.resolver,
        )
        self.f_loss_full = self.factory.make_jit_loss_full()
        self.f_loss = self.factory.make_jit_loss()
        self.f_grad = self.factory.make_grad()
        self.f_loss_vmap = self.factory.make_vmap_loss()

    def __call__(self, x):
        values = self.f_loss_full(jnp.array(x))
        loss = float(jnp.sum(values))
        self.constr = np.asarray(values[1:]).tolist()
        return loss

    def constraints(self):
        return self.constr


def get_lens_opt_problem():
    prob = lens_opt()
    ioh.problem.wrap_real_problem(prob, name='lens_opt',
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem('lens_opt', dimension=prob.dim)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    return problem
