# fmt: off
import os
import sys
import torch
import warnings
import numpy as np
sys.path.insert(0, os.getcwd())
from ioh import logger
from LLaMEA.llamea import LLaMEA, OpenAI_LLM
from LLaMEA.misc import aoc_logger, correct_aoc, OverBudgetException
from utils.extract_top_funcs import extract_top_funcs
from utils.extract_top_bbob import extract_top_bbob
from utils.problems_factory import ProblemName, get_example_problem
# fmt: on

warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_y_bounds(problem):
    x = np.random.uniform(problem.bounds.lb, problem.bounds.ub,
                          (1000*problem.meta_data.n_variables,
                           problem.meta_data.n_variables))
    y = problem(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return y_max + (y_max - y_min) * 0.2


################################################################################
problem_name = ProblemName.META_SURFACE
# exp_prefix = ""
# exp_prefix = "BBOB_"
# exp_prefix = "gp_func_"
exp_prefix = "xgboost_"
################################################################################


budget_cof = 100
real_problem = get_example_problem(problem_name)
problem_name_ioh = real_problem.meta_data.name
dim = real_problem.meta_data.n_variables
experiment_name = f"{exp_prefix}{problem_name_ioh}_{budget_cof}xD"


budget = budget_cof * dim
lb = real_problem.bounds.lb
ub = real_problem.bounds.ub
if experiment_name == f"gp_func_{problem_name_ioh}_{budget_cof}xD" or experiment_name == f"xgboost_{problem_name_ioh}_{budget_cof}xD":
    gp_problems = extract_top_funcs(
        gp_exp_path=f"data/GP_results/{problem_name_ioh}",
        dim=dim, real_lb=lb, real_ub=ub, nbest=3)
    gp_uppers = [find_y_bounds(problem) for problem in gp_problems]
elif experiment_name == f"{problem_name_ioh}_{budget_cof}xD":
    gp_problems = [real_problem]
    gp_uppers = [1e6]
elif experiment_name == f"BBOB_{problem_name_ioh}_{budget_cof}xD":
    gp_problems = extract_top_bbob(problem_name=problem_name_ioh, dim=dim)


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
llm = OpenAI_LLM(api_key, ai_model)
# Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
# Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
# CodeLlama-7b-Instruct-hf, CodeLlama-13b-Instruct-hf,
# CodeLlama-34b-Instruct-hf, CodeLlama-70b-Instruct-hf,


def evaluate_gp_func(solution, explogger=None, details=False):
    auc_mean = 0
    auc_std = 0
    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())
    aucs = []
    algorithm = None
    l2 = aoc_logger(budget, upper=max(gp_uppers),
                    triggers=[logger.trigger.ALWAYS])
    for i in range(len(gp_problems)):
        problem = gp_problems[i]
        problem.attach_logger(l2)
        l2.reset(problem)
        problem.reset()
        for rep in range(3):
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

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.10f} with standard deviation {auc_std:0.10f}."

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution


with open(f"data/description/{problem_name_ioh}.txt", "r") as f:
    description = f.read()
task_prompt = f"""
The optimization algorithm should handle a wide range of tasks, which is evaluated on the similar problems of a real-world problem. 
{description}
Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between func.bounds.lb (lower bound) and func.bounds.ub (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
"""
print(task_prompt)

for experiment_i in range(5):
    # A 1+1 strategy
    es = LLaMEA(
        evaluate_gp_func,
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
