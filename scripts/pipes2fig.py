# fmt: off
import os
import sys
import cv2
import ioh
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from problems.meta_surface.problem import get_meta_surface_problem
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.create_pset import *
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import runParallelFunction, read_pickle
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr
from LLaMEA.misc import aoc_logger, correct_aoc, OverBudgetException
from LLaMEA.llamea import LLaMEA, OpenAI_LLM
from test_alg import DE_PSO_Optimizer
# fmt: on


if torch.cuda.is_available():
    print(
        f"CUDA is available. PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Using CPU.")
torch.cuda.empty_cache()
# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gpt-4o"
experiment_name = "meta_surface_gp"
llm = OpenAI_LLM(api_key, ai_model)
# Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
# Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
# CodeLlama-7b-Instruct-hf, CodeLlama-13b-Instruct-hf,
# CodeLlama-34b-Instruct-hf, CodeLlama-70b-Instruct-hf,

evaluation_id = 0


def test_func1(x):
    x = np.clip(x, 0, 0)
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(mul(7.040400702545974, 4.70841758564605)),
            4.70841758564605)), x), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.7


def test_func2(x):
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(mul(7.040400702545974, 4.70841758564605)), 4.70841758564605)), cos(sub(sub(prod_vec(prod_vec(mul(prod_vec(mul(
        7.040400702545974, 4.70841758564605)), 4.70841758564605))), x), prod_vec(prod_vec(prod_vec(2.337443560678286)))))), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def test_func3(x):
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(2.337443560678286), 4.70841758564605)),
            x), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def test_func4(x):
    y = neg(cos(sub(sub(prod_vec(cos(square(1.0156711080086))), x),
            prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def test_meta1(x):
    y = sub(add(div(sub(1.0686808226656992, 5.043220782279738), mul(x, x)), sqrt(sub(4.852820708841654, sub(reciprocal(4.852820708841654), x)))), div(mul(sum_vec(x), sub(mul(sum_vec(x), sub(9.233098180677409, sub(reciprocal(div(mul(sum_vec(x), sub(
        5.043220782279738, 9.233098180677409)), 4.852820708841654)), 5.043220782279738))), sub(sub(reciprocal(4.852820708841654), sub(9.233098180677409, sub(reciprocal(roundoff(exp(x))), 5.043220782279738))), 5.043220782279738))), 4.852820708841654))
    return np.mean(y)+1e16


def test_meta2(x):
    y = sub(add(div(sub(1.0686808226656992, 5.043220782279738), mul(x, x)), sub(reciprocal(roundoff(exp(x))), 5.043220782279738)), div(mul(sum_vec(x), sub(mul(sum_vec(x), sub(9.233098180677409, sub(reciprocal(div(mul(sum_vec(x), sub(
        5.043220782279738, 9.233098180677409)), 4.852820708841654)), 5.043220782279738))), sub(sub(reciprocal(4.852820708841654), sub(9.233098180677409, sub(reciprocal(roundoff(exp(x))), 5.043220782279738))), 5.043220782279738))), 4.852820708841654))
    return np.mean(y)+1e16


def test_meta3(x):
    y = sub(add(div(sub(1.0686808226656992, 5.043220782279738), mul(x, x)), sqrt(sub(4.852820708841654, sub(reciprocal(4.852820708841654), 5.043220782279738)))), div(mul(sum_vec(x), sub(mul(sum_vec(x), sub(9.233098180677409, sub(reciprocal(div(mul(sum_vec(
        x), sub(5.043220782279738, 9.233098180677409)), 4.852820708841654)), 5.043220782279738))), sub(sub(reciprocal(5.043220782279738), sub(9.233098180677409, sub(reciprocal(roundoff(exp(x))), 5.043220782279738))), 5.043220782279738))), 4.852820708841654))
    return np.mean(y)+1e16


def test_meta4(x):
    y = sub(add(div(sub(1.0686808226656992, 5.043220782279738), mul(x, x)), sqrt(sub(4.852820708841654, sub(reciprocal(4.852820708841654), 5.043220782279738)))), div(mul(sum_vec(x), sub(mul(sum_vec(x), sub(9.233098180677409, sub(reciprocal(div(mul(sum_vec(
        x), sub(5.043220782279738, 9.233098180677409)), 4.852820708841654)), 5.043220782279738))), sub(sub(reciprocal(4.852820708841654), sub(9.233098180677409, sub(reciprocal(roundoff(exp(x))), 5.043220782279738))), 5.043220782279738))), 4.852820708841654))
    return np.mean(y)+1e16


def test_meta5(x):
    y = sub(add(div(sub(1.0686808226656992, 5.043220782279738), mul(x, x)), sqrt(sub(cos(square(1.0671155633616218)), sub(reciprocal(4.852820708841654), 5.043220782279738)))), div(mul(sum_vec(x), sub(mul(sum_vec(x), sub(9.233098180677409, sub(reciprocal(div(mul(
        sum_vec(x), sub(5.043220782279738, 9.233098180677409)), 4.852820708841654)), 5.043220782279738))), sub(sub(reciprocal(4.852820708841654), sub(9.233098180677409, sub(reciprocal(roundoff(exp(x))), 5.043220782279738))), 5.043220782279738))), 4.852820708841654))
    return np.mean(y)+1e16


class LlaMEA_topology:
    def __init__(self):
        self.evaluations = 0

    def __call__(self, solution, explogger=None, details=False):
        self.evaluations += 1
        auc_mean = 0
        auc_std = 0
        code = solution.code
        algorithm_name = solution.name
        exec(code, globals())
        error = ""
        aucs = []
        algorithm = None
        problems = []
        if self.evaluations % 20 == 0:
            problems += [get_pipes_topology_problem()]
        else:
            ioh.problem.wrap_real_problem(test_func1,
                                          name="func_gp1",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp1", dimension=23)]
            ioh.problem.wrap_real_problem(test_func2,
                                          name="func_gp2",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp2", dimension=23)]
            ioh.problem.wrap_real_problem(test_func3,
                                          name="func_gp3",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp3", dimension=23)]
            ioh.problem.wrap_real_problem(test_func4,
                                          name="func_gp4",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp4", dimension=23)]
        problems = [get_pipes_topology_problem()]
        dim = problems[0].meta_data.n_variables
        budget = 1000 * dim
        for i in range(len(problems)):
            problem = problems[i]
            l2 = aoc_logger(budget, lower=1e-8, upper=1.4,
                            triggers=[ioh.logger.trigger.ALWAYS])
            problem.attach_logger(l2)
            for rep in range(3):
                np.random.seed(rep)
                try:
                    algorithm = globals()[algorithm_name](
                        budget=budget, dim=dim)
                    algorithm(problem)
                except OverBudgetException:
                    pass
                auc = correct_aoc(problem, l2, budget)
                aucs.append(auc)
                l2.reset(problem)
                problem.reset()
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        i = 0
        while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
            i += 1
        np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)
        feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
        # if details:
        #     feedback = (
        #         f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Separable functions was {detailed_aucs[0]:.02f}, "
        #     )
        # feedback = ""
        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)
        return solution


class LlaMEA_meta_surface:
    def __init__(self):
        self.evaluations = 0

    def __call__(self, solution, explogger=None, details=False):
        self.evaluations += 1
        auc_mean = 0
        auc_std = 0
        code = solution.code
        algorithm_name = solution.name
        exec(code, globals())
        error = ""
        aucs = []
        algorithm = None
        problems = []
        uppers = []
        if self.evaluations % 10 == 0:
            problems += [get_meta_surface_problem()]
            uppers += [0.6]
        else:
            ioh.problem.wrap_real_problem(test_meta1,
                                          name="func_gp1",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp1", dimension=20)]
            ioh.problem.wrap_real_problem(test_meta2,
                                          name="func_gp2",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp2", dimension=20)]
            ioh.problem.wrap_real_problem(test_meta3,
                                          name="func_gp3",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp3", dimension=20)]
            ioh.problem.wrap_real_problem(test_meta4,
                                          name="func_gp4",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp4", dimension=20)]
            ioh.problem.wrap_real_problem(test_meta5,
                                          name="func_gp5",
                                          optimization_type=ioh.OptimizationType.MIN,)
            problems += [ioh.get_problem("func_gp5", dimension=20)]
            meta_problem = get_meta_surface_problem()
            for problem in problems:
                x = np.random.uniform(meta_problem.bounds.lb, meta_problem.bounds.ub, (10000, 20))
                y = problem(x)
                uppers += [(np.max(y) - np.min(y)) * 1.1]
        # problems = [get_meta_surface_problem()]
        # uppers = [0.6]
        dim = problems[0].meta_data.n_variables
        budget = 1000 * dim
        for i in range(len(problems)):
            problem = problems[i]
            l2 = aoc_logger(budget, lower=1e-8, upper=uppers[i],
                            triggers=[ioh.logger.trigger.ALWAYS])
            problem.attach_logger(l2)
            for rep in range(3):
                np.random.seed(rep)
                try:
                    algorithm = globals()[algorithm_name](
                        budget=budget, dim=dim)
                    algorithm(problem)
                except OverBudgetException:
                    pass
                auc = correct_aoc(problem, l2, budget)
                aucs.append(auc)
                l2.reset(problem)
                problem.reset()
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        i = 0
        while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
            i += 1
        np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)
        if self.evaluations % 10 == 0:
            feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f} on the real problem."
        else:
            feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f} on similar problems with similar landscape features."
        print(algorithm_name, algorithm, auc_mean, auc_std)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, feedback)
        return solution


if __name__ == "__main__":
    # df = pd.read_csv("data/GP_results/meta_surface_pca/gpfg_opt_runs.csv")
    # sorted_df = df.drop_duplicates(
    #     subset=[df.columns[2]]).sort_values(by=df.columns[2])
    # counter = 0
    # for index, row in sorted_df.iterrows():
    #     if counter < 5:
    #         print(
    #             f"index {index} | fitness: {row.iloc[2]} | formula: {row.iloc[3]}")
    #         print("-" * 100)
    #         counter += 1
    # array = np.load("data/y/meta_surface_pca.npy")
    # print(np.max(array), np.min(array))
    evaluate_demo = LlaMEA_meta_surface()
    task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on a real-world problem, meta-surface design. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
"""
    for experiment_i in range(5):
        # A 1+1 strategy
        es = LLaMEA(
            evaluate_demo,
            llm=llm,
            n_parents=1,
            n_offspring=1,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=100,
        )
        print(es.run())

#     alg = DE_PSO_Optimizer(dim*1000, dim)
#     alg(problem)
