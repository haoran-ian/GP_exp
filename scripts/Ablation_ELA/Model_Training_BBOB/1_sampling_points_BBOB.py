# fmt: off
import os
import sys
import ioh
import warnings
import numpy as np
sys.path.insert(0, os.getcwd())
from pflacco.sampling import create_initial_sample
# fmt: on

warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == "__main__":
    blocks = 3
################################################################################
    fid = int(sys.argv[1])
    iid = int(sys.argv[2])
    dim = int(sys.argv[3])
    seed = int(sys.argv[4])
    block_coef = float(sys.argv[5])
    n_coef = int(sys.argv[6])
################################################################################
    file_name = f"F{fid}-{iid}-D{dim}-seed:{seed}-block_coef:{block_coef:.1f}-n_coef:{n_coef}.npy"
    if os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
        sys.exit()
    print(f"Creating {file_name} ...")
    problem = ioh.get_problem(fid=fid, instance=iid, dimension=dim,
                              problem_class=ioh.ProblemClass.BBOB)
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    if block_coef != 0:
        n_sample = int(dim**blocks * 3 * block_coef)
    else:
        n_sample = dim * n_coef
    X = create_initial_sample(dim=dim, n=n_sample, lower_bound=lb,
                              upper_bound=ub, sample_type='sobol', seed=seed)
    y = np.array(problem(X.values))
    np.save(f"data/Ablation_ELA/X/{file_name}", X)
    np.save(f"data/Ablation_ELA/Y/{file_name}", y)
