# fmt: off
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.getcwd())
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.sampling import sampling
from problems.meta_surface.problem import get_meta_surface_problem
from problems.fluid_dynamics.problem import get_pipes_topology_problem
# fmt: on

if __name__ == "__main__":
    problem = get_pipes_topology_problem(0, 3)
    y = np.load("data/y/topology_3pipes_23D_instance0.npy")
    # problem = get_meta_surface_problem()
    y = np.load("data/y/meta_surface.npy")
    dim = problem.meta_data.n_variables
    ndoe = 150 * dim
    doe_x = sampling('lhs', n=ndoe, lower_bound=problem.bounds.lb,
                     upper_bound=problem.bounds.ub, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    # y = problem(doe_x)
    # y = np.array(y)
    start_time = time.time()
    df_ela = bootstrap_ela(doe_x, y, bs_ratio=0.8, bs_repeat=5,
                           lower_bound=problem.bounds.lb,
                           upper_bound=problem.bounds.ub)
    print(f"bootstrap_ela time; {time.time()-start_time}s")
    print(df_ela)
