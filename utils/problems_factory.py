# fmt: off
import os
import sys
sys.path.insert(0, os.getcwd())
from enum import Enum
# fmt: on


class ProblemName(Enum):
    META_SURFACE = 0
    META_SURFACE_SOLVER = 1
    PHOTONIC_2LAYERS_ELLIPSOMETRY = 2
    PHOTONIC_10LAYERS_BRAGG = 3
    PHOTONIC_20LAYERS_BRAGG = 4
    PHOTONIC_10LAYERS_PHOTOVOLTAIC = 5


def get_example_problem(problem_name: ProblemName):
    if problem_name == ProblemName.META_SURFACE:
        from problems.meta_surface.problem import get_meta_surface_problem
        return get_meta_surface_problem()
    elif problem_name == ProblemName.PHOTONIC_2LAYERS_ELLIPSOMETRY:
        from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
        return get_photonic_problem(problem_type=PROBLEM_TYPE.ELLIPSOMETRY)
    elif problem_name == ProblemName.PHOTONIC_10LAYERS_BRAGG:
        from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
        return get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.BRAGG)
    elif problem_name == ProblemName.PHOTONIC_20LAYERS_BRAGG:
        from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
        return get_photonic_problem(num_layers=20, problem_type=PROBLEM_TYPE.BRAGG)
    elif problem_name == ProblemName.PHOTONIC_10LAYERS_PHOTOVOLTAIC:
        from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
        return get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.PHOTOVOLTAIC)
