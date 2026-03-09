# fmt: off
import os
import sys
import ioh
import numpy as np
sys.path.insert(0, os.getcwd())
from utils.problems_factory import ProblemName, get_example_problem
from gp_fgenerator.sampling import sampling
# fmt: on


if __name__ == '__main__':
    coef_ela = 1000
    problems = [
        get_example_problem(ProblemName.META_SURFACE),
    ]
    for problem in problems:
        dim = problem.meta_data.n_variables
        name = problem.meta_data.name
        print(name)
        print(problem.bounds.lb)
        print(problem.bounds.ub)
        ndoe = coef_ela*dim
        doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
                         upper_bound=problem.bounds.ub, round_off=2,
                         random_seed=42, verbose=True).create_doe()
        if not os.path.exists(f'data/1000D_y/{name}.npy'):
            l1 = ioh.logger.Analyzer(
                folder_name=f'data/1000D_y_ioh_dat/{name}',
                triggers=[ioh.logger.trigger.ALWAYS],
                store_positions=True
            )
            problem.attach_logger(l1)
            y = problem(doe_x)
            np.save(f'data/1000D_y/{name}.npy', y)
        y = np.load(f'data/1000D_y/{name}.npy')
