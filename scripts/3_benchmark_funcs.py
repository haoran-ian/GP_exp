# fmt: off
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
from utils.extract_top_funcs import extract_top_funcs
from problems.fluid_dynamics.problem import get_pipes_topology_problem
# fmt: on

if __name__ == "__main__":
    iid = 0
    num_pipes = 3
    gp_exp = f"data/GP_results/fluid_dynamics_{num_pipes}pipes_iid{iid}"
    problem = get_pipes_topology_problem(iid=iid, num_pipes=num_pipes)
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    cheap_problems = extract_top_funcs(gp_exp, dim, lb, ub)
    x = np.random.uniform(lb, ub, (10, lb.shape[0]))
    print(lb)
    print(cheap_problems[0](x))
