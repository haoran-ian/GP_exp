# fmt: off
import os
import sys
import warnings
import pandas as pd
sys.path.insert(0, os.getcwd())
# from problems.fluid_dynamics.problem import get_pipes_topology_problem
# from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import get_photonic_problem, PROBLEM_TYPE
from gp_fgenerator.sampling import sampling
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import read_pickle
# fmt: on

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


if __name__ == "__main__":
    # exp_params =
    # [10, PROBLEM_TYPE.BRAGG, "photonic_10layers_bragg"],
    # exp_params = [20, PROBLEM_TYPE.BRAGG, "photonic_20layers_bragg"]
    # [1, PROBLEM_TYPE.ELLIPSOMETRY, "photonic_2layers_ellipsometry"],
    # [10, PROBLEM_TYPE.PHOTOVOLTAIC, "photonic_10layers_photovoltaic"],
    exp_params = [20, PROBLEM_TYPE.PHOTOVOLTAIC,
                  "photonic_20layers_photovoltaic"]
    # for param in exp_params:
    problem_label = exp_params[2]
    path = f"data/ELA/ela_{problem_label}"
    problem = get_photonic_problem(num_layers=exp_params[0],
                                   problem_type=exp_params[1])
    dim = problem.meta_data.n_variables
    print(dim)
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
                     upper_bound=problem.bounds.ub, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    target_vector = pd.read_csv(f"{path}/ela_60.csv").mean()
    list_ela = read_pickle(f"{path}/ela_corr.pickle")
    ela_min = read_pickle(f"{path}/ela_min.pickle")
    ela_max = read_pickle(f"{path}/ela_max.pickle")
    dist_metric = "cityblock"
    r = 1
    GP_fgen = GP_func_generator(doe_x,
                                target_vector,
                                bs_ratio=0.8,
                                bs_repeat=5,
                                list_ela=list_ela,
                                ela_min=ela_min,
                                ela_max=ela_max,
                                ela_weight={},
                                dist_metric=dist_metric,
                                problem_label=problem_label,
                                filepath_save=f'data/GP_results/{problem_label}',
                                tree_size=(3, 12),
                                population=50,
                                cxpb=0.5,
                                mutpb=0.1,
                                ngen=50,
                                nhof=1,
                                seed=r,
                                verbose=True)
    hof, pop = GP_fgen()
