# fmt: off
import os
import sys
import warnings
import numpy as np
sys.path.insert(0, os.getcwd())
from pflacco.sampling import create_initial_sample
from utils.problems_factory import ProblemName, get_example_problem
# fmt: on

warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == "__main__":
    blocks = 3
################################################################################
    problem_name = ProblemName(int(sys.argv[1]))
    seed = int(sys.argv[2])
    block_coef = float(sys.argv[3])
    n_coef = int(sys.argv[4])
################################################################################
    file_name = f"{problem_name}-seed:{seed}-block_coef:{block_coef}-n_coef:{n_coef}.npy"
    if os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
        sys.exit()
    print(f"Creating {file_name} ...")
    problem = get_example_problem(problem_name)
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    if block_coef != 0:
        n_sample = int(dim**blocks * 3 * block_coef)
    else:
        n_sample = dim * n_coef
    X = create_initial_sample(dim=dim, n=n_sample, lower_bound=lb,
                              upper_bound=ub, sample_type='lhs', seed=seed)
    y = np.array(problem(X.values))
    np.save(f"data/Ablation_ELA/X/{file_name}", X)
    np.save(f"data/Ablation_ELA/Y/{file_name}", y)
