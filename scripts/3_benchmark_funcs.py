# fmt: off
import os
import sys
import ioh
import numpy as np
sys.path.insert(0, os.getcwd())
from sko.DE import DE
from utils.extract_top_funcs import extract_top_funcs
from problems.fluid_dynamics.problem import get_pipes_topology_problem
# fmt: on


def benchmark(problem: ioh.ProblemClass.REAL, exp_name: str, runs: int = 20, seed=42):
    np.random.seed(seed)
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    budgets = [i * dim for i in [10, 50, 100]]
    for budget in budgets:
        if budget % 50 > 0:
            size_pop = 10
        else:
            size_pop = 50
        l1 = ioh.logger.Analyzer(
            folder_name=f"data/benchmark_funcs/{exp_name}_DE_{int(budget/dim)}",
            algorithm_name="DE",
            triggers=[ioh.logger.trigger.ALWAYS],
            store_positions=True
        )
        problem.attach_logger(l1)
        for _ in range(runs):
            de = DE(func=problem, n_dim=dim, size_pop=size_pop,
                    max_iter=int(budget/size_pop/2), lb=lb, ub=ub)
            de.run()
            problem.reset()
        l1.close()


if __name__ == "__main__":
    iid = 0
    num_pipes = 3
    gp_exp = f"data/GP_results/fluid_dynamics_{num_pipes}pipes_iid{iid}"
    real_problem = get_pipes_topology_problem(iid=iid, num_pipes=num_pipes)
    dim = real_problem.meta_data.n_variables
    lb = real_problem.bounds.lb
    ub = real_problem.bounds.ub
    # cheap_problems = extract_top_funcs(gp_exp, dim, lb, ub)
    # for cheap_p in cheap_problems:
    #     benchmark(
    #         cheap_p, f"{cheap_p.meta_data.name}_{real_problem.meta_data.name}")
    for fid in range(1, 25):
        bbob_problem = ioh.get_problem(fid=fid, instance=0, dimension=dim,
                                       problem_class=ioh.ProblemClass.BBOB)
        benchmark(bbob_problem, f"BBOB_F{fid}_D{dim}")
    benchmark(real_problem, real_problem.meta_data.name)
