# fmt: off
import os
import sys
import ioh
import warnings
import numpy as np
import nevergrad as ng
import niapy.algorithms as nialg
import opytimizer.optimizers as opy_opt
import opytimizer.optimizers.swarm as swarm
import opytimizer.optimizers.science as science
import opytimizer.optimizers.social as social
import opytimizer.optimizers.population as population
import opytimizer.optimizers.evolutionary as evolutionary
import opytimizer.optimizers.misc as misc
sys.path.insert(0, os.getcwd())
from mealpy_helper import get_models
from niapy.problems import Problem
from niapy.task import Task
from opytimizer.spaces import SearchSpace
from opytimizer.core import Function
from opytimizer import Opytimizer
from modcma import ModularCMAES
from modde import ModularDE
from copy import deepcopy, copy
from multiprocessing import Pool
from itertools import product
from utils.problems_factory import ProblemName, get_example_problem
# fmt: on

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


DATA_FOLDER = "Benchmark_Top_Results"
MAX_THREADS = 32

TOP_ALGORITHMS = {
    # Baselines (modcma, modde, nevergrad)
    "modcma_bipop": "baseline",
    "modcma_base": "baseline",
    "modde_lshade": "baseline",
    "modde_base": "baseline",
    "ng_MultiBFGS": "baseline",
    "ng_Powell": "baseline",
    "ng_DiagonalCMA": "baseline",
    "ng_OnePlusOne": "baseline",
    "ng_DE": "baseline",
    # Opytimizer
    "opy_CS": "opytimizer",
    "opy_DE": "opytimizer",
    "opy_WEO": "opytimizer",
    "opy_QSA": "opytimizer",
    "opy_COA": "opytimizer",
    "opy_SOS": "opytimizer",
    "opy_AEO": "opytimizer",
    "opy_ABO": "opytimizer",
    "opy_GCO": "opytimizer",
    "opy_LSA": "opytimizer",
    "opy_RRA": "opytimizer",
    "opy_SAVPSO": "opytimizer",
    "opy_IWO": "opytimizer",
    "opy_EO": "opytimizer",
    "opy_SSA": "opytimizer",
    "opy_BSA": "opytimizer",
    "opy_ABC": "opytimizer",
    "opy_GOA": "opytimizer",
    "opy_GA": "opytimizer",
    "opy_TWO": "opytimizer",
    "opy_RPSO": "opytimizer",
    "opy_RFO": "opytimizer",
    "opy_ASO": "opytimizer",
    "opy_AOA": "opytimizer",
    "opy_MVPA": "opytimizer",
    # Mealpy
    "mealpy_JADE": "mealpy",
    "mealpy_SADE": "mealpy",
    "mealpy_L_SHADE": "mealpy",
    "mealpy_OriginalSARO": "mealpy",
    "mealpy_OriginalTWO": "mealpy",
    "mealpy_LevyES": "mealpy",
    "mealpy_BaseVCS": "mealpy",
}

# 1. Baseline Evaluator
modde_params = {
    'base': {'mutation_base': 'rand', 'mutation_reference': None, 'lpsr': False,
             'lambda_': 10, 'use_archive': False},
    'lshade': {'mutation_base': 'target', 'mutation_reference': 'pbest',
               'lpsr': True, 'lambda_': 18, 'use_archive': True,
               'adaptation_method_F': 'shade', 'adaptation_method_CR': 'shade'}
}
modcma_params = {'base': {}, 'bipop': {'local_restart': 'BIPOP'}}


class Evaluator_Baseline():
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):
        for seed in range(n_reps):
            np.random.seed(int(seed))
            budget = int(100 * func.meta_data.n_variables)
            if self.alg.startswith('ng_'):
                parametrization = ng.p.Array(
                    shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
                optimizer = eval(f"ng.optimizers.{self.alg[3:]}")(
                    parametrization=parametrization, budget=budget)
                optimizer.minimize(func)
            elif self.alg.startswith('modde_'):
                params = deepcopy(modde_params[self.alg[6:]])
                params['lambda_'] *= func.meta_data.n_variables
                c = ModularDE(func, bound_correction='saturate',
                              budget=budget, **params)
                c.run()
            else:  # modcma
                params = modcma_params[self.alg[7:]]
                c = ModularCMAES(func, d=func.meta_data.n_variables,
                                 bound_correction='saturate', budget=budget,
                                 x0=np.zeros((func.meta_data.n_variables, 1)),
                                 **params)
                c.run()
            func.reset()

# 2. Opytimizer Evaluator


class Evaluator_Opytimizer():
    def __init__(self, optimizer):
        self.alg_name = optimizer.replace("opy_", "")

    def __call__(self, func, n_reps):
        def helper(x):
            return func(x.reshape(-1))
        modules = [swarm, science, social, population, evolutionary, misc]
        opt_class = None
        for module in modules:
            if hasattr(module, self.alg_name):
                opt_class = getattr(module, self.alg_name)
                break
        if not opt_class:
            print(
                f"Skipping {self.alg_name}, not found in Opytimizer modules.")
            return
        space = SearchSpace(n_agents=30,
                            n_variables=func.meta_data.n_variables,
                            lower_bound=func.bounds.lb,
                            upper_bound=func.bounds.ub)
        optimizer_instance = opt_class()
        function = Function(helper)
        for seed in range(n_reps):
            np.random.seed(int(seed))
            opt_runner = Opytimizer(space, optimizer_instance, function)
            n_iter = int((func.meta_data.n_variables * 100) / 30)
            opt_runner.start(n_iterations=n_iter)
            func.reset()

# 3. Mealpy Evaluator


class Evaluator_Mealpy():
    def __init__(self, optimizer_name):
        self.alg_name = optimizer_name.replace("mealpy_", "")
        self.models = get_models()

    def __call__(self, func, n_reps):
        term = {"max_fe": func.meta_data.n_variables * 100}
        problem = {"fit_func": func, "lb": func.bounds.lb, "ub": func.bounds.ub,
                   "minmax": "min", "log_to": None, "save_population": False}
        model = next((m for m in self.models if m.name == self.alg_name), None)
        if not model:
            print(
                f"Skipping {self.alg_name}, not configured in mealpy_helper.py.")
            return

        for seed in range(n_reps):
            np.random.seed(int(seed))
            model_copy = copy(model)
            model_copy.solve(problem, termination=term)
            func.reset()


def run_unified_optimizer(temp):
    alg_key, pid = temp
    lib_source = TOP_ALGORITHMS[alg_key]
    problem = get_example_problem(ProblemName(pid))
    dim = problem.meta_data.n_variables
    print(f"Running: {alg_key} from {lib_source} | {ProblemName(pid)}")
    if lib_source == "baseline":
        algorithm = Evaluator_Baseline(alg_key)
    elif lib_source == "opytimizer":
        algorithm = Evaluator_Opytimizer(alg_key)
    elif lib_source == "mealpy":
        algorithm = Evaluator_Mealpy(alg_key)
    else:
        return

    folder_path = f"{DATA_FOLDER}/{lib_source.upper()}/"
    os.makedirs(folder_path, exist_ok=True)

    logger = ioh.logger.Analyzer(
        root=folder_path, folder_name=f"{alg_key}_{ProblemName(pid)}", algorithm_name=f"{alg_key}")

    problem.attach_logger(logger)
    try:
        algorithm(problem, 5)
    except Exception as e:
        print(f"Error running {alg_key} on {ProblemName(pid)}: {e}")
    finally:
        logger.close()


def runParallelFunction(runFunction, arguments):
    arguments = list(arguments)
    p = Pool(min(MAX_THREADS, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    algnames = list(TOP_ALGORITHMS.keys())
    pids = range(2, 6)
    args = product(algnames, pids)

    print(f"Total combinations to run: {len(algnames) * len(pids)}")
    runParallelFunction(run_unified_optimizer, args)
