import ioh
import numpy as np
from enum import Enum
from problems.photovotaic_problems.brag_mirror import brag_mirror
from problems.photovotaic_problems.ellipsometry_inverse import ellipsometry
from problems.photovotaic_problems.sophisticated_antireflection_design import sophisticated_antireflection_design


class PROBLEM_TYPE(Enum):
    BRAGG = 0
    ELLIPSOMETRY = 1
    PHOTOVOLTAIC = 2


def get_photonic_problem(num_layers: int = 10,
                         problem_type: PROBLEM_TYPE = PROBLEM_TYPE.BRAGG):
    if problem_type == PROBLEM_TYPE.BRAGG:
        nb_layers = num_layers
        target_wl = 600.0  # nm
        mat_env = 1.0      # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(prob, name=f"photonic_{nb_layers}layers_bragg",
                                      optimization_type=ioh.OptimizationType.MIN,
                                      lb=prob.min_thick, ub=prob.max_thick)
        problem = ioh.get_problem(f"photonic_{nb_layers}layers_bragg", dimension=prob.n)
        return problem
    elif problem_type == PROBLEM_TYPE.ELLIPSOMETRY:
        mat_env = 1.0
        mat_substrate = 'Gold'
        nb_layers = 1
        min_thick = 50     # nm
        max_thick = 150
        min_eps = 1.1      # permittivity
        max_eps = 3
        wavelengths = np.linspace(400, 800, 100)  # nm
        angle = 40*np.pi/180  # rad
        prob = ellipsometry(mat_env, mat_substrate, nb_layers, min_thick, max_thick,
                            min_eps, max_eps, wavelengths, angle)
        ioh.problem.wrap_real_problem(prob, name="ellipsometry",
                                      optimization_type=ioh.OptimizationType.MIN,)
        problem = ioh.get_problem("ellipsometry", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        return problem
    elif problem_type == PROBLEM_TYPE.PHOTOVOLTAIC:
        nb_layers = num_layers
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                   wl_min, wl_max)
        ioh.problem.wrap_real_problem(prob, name=f"sophisticated_antireflection_design_{nb_layers}",
                                      optimization_type=ioh.OptimizationType.MIN,
                                      lb=prob.min_thick, ub=prob.max_thick)
        problem = ioh.get_problem(f"sophisticated_antireflection_design_{nb_layers}",
                                  dimension=prob.n)
        return problem
    else:
        print(f"PROBLEM_TYPE {problem_type} is not supported.")
        return None
