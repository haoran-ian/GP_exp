# fmt: off
import os
import sys
import ioh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from problems.lens_opt.problem import get_lens_opt_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.create_pset import *
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr
# fmt: on


if __name__ == '__main__':
    ############################################################################
    # num_pipes = 3
    # iid = 2
    # problem = get_pipes_topology_problem(iid=iid, num_pipes=num_pipes)
    # dim = problem.meta_data.n_variables
    # exp_name = f'topology_{num_pipes}pipes_{dim}D_instance{iid}'
    ############################################################################
    # exp_name = 'fluid_dynamics_3pipes_iid0'
    # problem = get_meta_surface_problem()
    # dim = problem.meta_data.n_variables
    ############################################################################
    # problem = get_photonic_problem(20, PROBLEM_TYPE.PHOTOVOLTAIC)
    # dim = problem.meta_data.n_variables
    # exp_name = f'photonic_{dim}layers_photovoltaic'
    ############################################################################
    # problem = get_lens_opt_problem()
    # dim = problem.meta_data.n_variables
    # exp_name = 'lens_opt'
    ############################################################################
    problem = get_meta_surface_problem()
    dim = problem.meta_data.n_variables
    exp_name = 'meta_surface'
    ############################################################################
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
                     upper_bound=problem.bounds.ub, round_off=2,
                     random_seed=42, verbose=True).create_doe()
    x = doe_x[42]
    x = np.where(x <= 0.5, 0., 1.)
    triangle = problem.vector_to_triangle(x)
    quarter_square = problem.reflect_triangle(triangle)
    square = problem.rotate_around_corner(quarter_square)
    x_in = problem.create_final_image(square).reshape((1, 1, 36, 36))
    sns.heatmap(x_in, square=True)
    plt.savefig("test.png")
    plt.close()
    
