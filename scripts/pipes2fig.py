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
experiment_name = "topology_3"
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


def evaluate_demo(solution, explogger=None, details=False):
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0, 0, 0, 0, 0]
    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())

    error = ""

    aucs = []
    # detail_aucs = []
    algorithm = None
    # evaluation_id = 1
    # if evaluation_id < 100:
    problems = []
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
    # else:
    #     rect_width = 2.0
    #     rect_height = 1.0
    #     num_pipes = 3
    #     # exp_name = f"topology_{num_pipes}_{}"
    #     start_points = []
    #     end_points = []
    #     for i in range(num_pipes):
    #         entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #         exit_point = (rect_width, np.random.uniform(
    #             0.1, 0.9) * rect_height)
    #         start_points += [entry_point]
    #         end_points += [exit_point]
    #     prob = pipes_topology(num_pipes, start_points, end_points)
    #     ioh.problem.wrap_real_problem(prob, name="topology",
    #                                   optimization_type=ioh.OptimizationType.MIN,)
    #     problem = ioh.get_problem("topology", dimension=prob.pca_dim)
    #     problem.bounds.lb = prob.pca_lb
    #     problem.bounds.ub = prob.pca_ub
    dim = problems[0].meta_data.n_variables
    budget = 1000 * dim
    for i in range(len(problems)):
        problem = problems[i]
        # problem = ioh.get_problem(16, 1, dim, problem_class=ioh.ProblemClass.BBOB)
        # lower = -0.7 if i==0 else -0.6
        l2 = aoc_logger(budget, lower=1e-8, upper=1.4,
                        triggers=[ioh.logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        for rep in range(5):
            np.random.seed(rep)
            try:
                algorithm = globals()[algorithm_name](budget=budget, dim=dim)
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


if __name__ == "__main__":
    rect_width = 2.0
    rect_height = 1.0
    # np.random.seed(42)
    # for num_pipes in range(3, 6):
    #     start_points = []
    #     end_points = []
    #     for i in range(num_pipes):
    #         entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #         exit_point = (rect_width, np.random.uniform(
    #             0.1, 0.9) * rect_height)
    #         start_points += [entry_point]
    #         end_points += [exit_point]

    #     problem = pipes_topology(num_pipes, start_points, end_points)
    #     dim = problem.dim
    #     X = np.random.uniform(problem.lb, problem.ub,
    #                           size=(1000*dim, problem.lb.shape[0]))

    #     threshold = 0.95
    #     optimal_n, cum_var = find_optimal_pca_components(X)
    #     print(
    #         f"Recommended reserved dimensions: {optimal_n} (keep {threshold*100}% covariance)")
    #     feature_importance_heatmap(X, title=f"cov_heatmap_{num_pipes}")

    # x = np.random.uniform(problem.pca_lb, problem.pca_ub)
    # y = problem(x)
    # print(y)

    # for num_pipes in range(3, 6):
    #     for iid in range(3):
    # num_pipes = int(sys.argv[1])
    # iid = int(sys.argv[2])
    # print(f"processing problem with {num_pipes} pipes with instance_id={iid}")
    # np.random.seed(42+num_pipes+iid)
    # start_points = []
    # end_points = []
    # for _ in range(num_pipes):
    #     entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #     exit_point = (rect_width, np.random.uniform(0.1, 0.9) * rect_height)
    #     start_points += [entry_point]
    #     end_points += [exit_point]
    # problem = get_pipes_topology_problem(iid, num_pipes)
    # start_time = time.time()
    # for _ in range(10):
    #     x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
    #     y = problem(x)
    #     print(y)
    # print(f"total {time.time()-start_time}s")
    # prob = pipes_topology(num_pipes, start_points, end_points)
    # ioh.problem.wrap_real_problem(prob, name="topology",
    #                             optimization_type=ioh.OptimizationType.MIN,)
    # problem = ioh.get_problem("topology", dimension=prob.pca_dim)
    # problem.bounds.lb = prob.pca_lb
    # problem.bounds.ub = prob.pca_ub
    # dim = problem.meta_data.n_variables
    # exp_name = f"topology_{num_pipes}pipes_{dim}D_instance{iid}"
    # if os.path.exists(f"data/ELA/{exp_name}"):
    #     exit()
    # ndoe = 150*dim
    # doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
    #                 upper_bound=problem.bounds.ub, round_off=2, random_seed=42,
    #                 verbose=True).create_doe()
    # if not os.path.exists(f"data/y/{exp_name}.npy"):
    #     l1 = ioh.logger.Analyzer(
    #         folder_name=exp_name,
    #         triggers=[ioh.logger.trigger.ALWAYS],
    #         store_positions=True
    #         )
    #     problem.attach_logger(l1)
    #     y = problem(doe_x)
    #     np.save(f"data/y/{exp_name}.npy", y)
    # else:
    #     y = np.load(f"data/y/{exp_name}.npy")
    # fid = problem.meta_data.problem_id
    # get_ela(dim, problem, doe_x, y, exp_name)
    # get_ela_normalize(dim, fid, exp_name)
    # get_ela_corr(dim, fid, exp_name)

    # target_vector = pd.read_csv(f"ela_{exp_name}/ela_1121.csv").mean()
    # list_ela = read_pickle(f"ela_{exp_name}/ela_corr.pickle")
    # ela_min = read_pickle(f"ela_{exp_name}/ela_min.pickle")
    # ela_max = read_pickle(f"ela_{exp_name}/ela_max.pickle")
    # dist_metric = "cityblock"
    # r = 1
    # GP_fgen = GP_func_generator(doe_x,
    #                             target_vector,
    #                             bs_ratio=0.8,
    #                             bs_repeat=5,
    #                             list_ela=list_ela,
    #                             ela_min=ela_min,
    #                             ela_max=ela_max,
    #                             ela_weight={},
    #                             dist_metric=dist_metric,
    #                             problem_label=f'regressor_p_diff_{dim}',
    #                             filepath_save=f'GP_results/regressor_p_diff_{dim}D_{dist_metric}_rep{r}',
    #                             tree_size=(3, 12),
    #                             population=50,
    #                             cxpb=0.5,
    #                             mutpb=0.1,
    #                             ngen=50,
    #                             nhof=1,
    #                             seed=r,
    #                             verbose=True)
    # hof, pop = GP_fgen()

    # ioh.problem.wrap_real_problem(test_func1,
    #                                   name="func_gptest",
    #                                   optimization_type=ioh.OptimizationType.MIN,)
    # problem = ioh.get_problem("func_gp1", dimension=23)
    # problem.bounds.lb = prob.pca_lb
    # problem.bounds.ub = prob.pca_ub
    # x = np.random.uniform(problem.bounds.lb, problem.bounds.ub, (100000,23))
    # y = problem(x)
    # print(np.min(y), np.max(y))
    task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
"""
    for experiment_i in range(3):
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
