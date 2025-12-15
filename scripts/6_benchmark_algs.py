# fmt: off
import os
import sys
import ioh
import numpy as np
sys.path.insert(0, os.getcwd())
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
from utils.extract_top_algs import extract_top_algs
from modcma import modularcmaes
from modde import ModularDE
# fmt: on


class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt


def benchmark_alg(solution, problem, budget, exp_name, runs=10):
    if os.path.exists(f'data/benchmark_algs/{exp_name}'):
        return
    code = solution.code
    algorithm_name = solution.name
    dim = problem.meta_data.n_variables
    exec(code, globals())
    l1 = ioh.logger.Analyzer(
        folder_name=f'data/benchmark_algs/{exp_name}',
        algorithm_name=algorithm_name,
        triggers=[ioh.logger.trigger.ALWAYS],
        store_positions=True
    )
    problem.attach_logger(l1)
    problem.reset()
    l1.reset()
    for _ in range(runs):
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        algorithm(problem)
        problem.reset()
        l1.reset()
    l1.close()


def benchmark_baseline(problem, budget, problem_name, runs=10):
    dim = problem.meta_data.n_variables
    RS = RandomSearch(budget=budget, dim=dim)
    algorithms = [RS, None, None, None]
    algorithm_names = ['RandomSearch', 'CMA-ES', 'DE', 'LSHADE']
    for i in range(len(algorithms)):
        if os.path.exists(f'data/benchmark_algs/{problem_name}/{algorithm_names[i]}_run0_best0'):
            continue
        l1 = ioh.logger.Analyzer(
            folder_name=f'data/benchmark_algs/{problem_name}/{algorithm_names[i]}_run0_best0',
            algorithm_name=algorithm_names[i],
            triggers=[ioh.logger.trigger.ALWAYS],
            store_positions=True
        )
        problem.attach_logger(l1)
        problem.reset()
        l1.reset()
        for _ in range(runs):
            if i == 0:
                algorithm = algorithms[i]
                algorithm(problem)
            elif i == 1:
                modularcmaes.fmin(
                    func=problem, x0=np.zeros(dim), budget=budget)
            elif i == 2:
                lshade = ModularDE(problem, budget=budget)
                lshade.run()
            elif i == 3:
                lshade = ModularDE(problem, budget=budget,
                                   base_sampler='uniform',
                                   mutation_base='target',
                                   mutation_reference='pbest',
                                   bound_correction='expc_center',
                                   crossover='bin', lpsr=True, lambda_=18*5,
                                   memory_size=6, use_archive=True,
                                   init_stats=True, adaptation_method_F='shade',
                                   adaptation_method_CR='shade')
                lshade.run()
            problem.reset()
            l1.reset()
        l1.close()


def extract_exp_paths_from_name(exp_name: str, problem_name: str,
                                LLaMEA_exp_root: str = 'data/LLaMEA_exp'):
    exp_paths = []
    parent_folder = problem_name
    if 'BBOB' in exp_name:
        parent_folder = 'LLaMEA_BBOB'
    folders = os.listdir(os.path.join(LLaMEA_exp_root, parent_folder))
    for folder in folders:
        if folder.split('-')[-1] == exp_name:
            exp_paths += [os.path.join(LLaMEA_exp_root, parent_folder, folder)]
    return exp_paths


if __name__ == '__main__':
    budget_cof = 10
    # problem_name = 'meta_surface'
    problem_name = 'photonic_10layers_bragg'
    problem = get_photonic_problem(
        num_layers=10, problem_type=PROBLEM_TYPE.BRAGG)
    # problem = get_meta_surface_problem()
    dim = problem.meta_data.n_variables
    if not os.path.exists(f'data/benchmark_algs/{problem_name}'):
        os.mkdir(f'data/benchmark_algs/{problem_name}')
    exp_names = [
        # f'BBOB_{10}xD',
        f'{problem_name}_{budget_cof}xD',
        f'gp_func_{problem_name}_{budget_cof}xD',
    ]
    for exp_name in exp_names:
        exp_paths = extract_exp_paths_from_name(exp_name=exp_name,
                                                problem_name=problem_name)
        for i in range(len(exp_paths)):
            exp_path = exp_paths[i]
            solutions = extract_top_algs(exp_path)
            for j in range(len(solutions)):
                exp_name_detailed = f'{problem_name}/{exp_name}_run{i}_best{j}'
                solution = solutions[j]
                benchmark_alg(solution=solution, problem=problem,
                              budget=budget_cof*dim, exp_name=exp_name_detailed)
    benchmark_baseline(problem=problem, budget=budget_cof*dim,
                       problem_name=problem_name)
