# fmt: off
import os
import sys
import warnings
import numpy as np
import pandas as pd
import nevergrad as ng
import opytimizer.optimizers.swarm as swarm
import opytimizer.optimizers.science as science
import opytimizer.optimizers.social as social
import opytimizer.optimizers.population as population
import opytimizer.optimizers.evolutionary as evolutionary
import opytimizer.optimizers.misc as misc
from copy import deepcopy, copy
from multiprocessing import Pool
from itertools import product
from types import SimpleNamespace
sys.path.insert(0, os.getcwd())
from opytimizer.spaces import SearchSpace
from opytimizer.core import Function
from opytimizer import Opytimizer
from modcma import ModularCMAES
from modde import ModularDE
from utils.mealpy_helper import get_models
from utils.extract_generated_problems import extract_llm_generated_problems
# fmt: on

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


DATA_FOLDER = "Benchmark_Top_Results_LLM"
INDEX_FILE = "data/LLM/llm_generated_problem_index.csv"
MAX_THREADS = 32
N_REPS = 5
BUDGET_FACTOR = 100

DEFAULT_DIM = 5
DEFAULT_LOWER_BOUND = -5.0
DEFAULT_UPPER_BOUND = 5.0


TOP_ALGORITHMS = {
    # Baselines
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


modde_params = {
    "base": {
        "mutation_base": "rand",
        "mutation_reference": None,
        "lpsr": False,
        "lambda_": 10,
        "use_archive": False,
    },
    "lshade": {
        "mutation_base": "target",
        "mutation_reference": "pbest",
        "lpsr": True,
        "lambda_": 18,
        "use_archive": True,
        "adaptation_method_F": "shade",
        "adaptation_method_CR": "shade",
    },
}

modcma_params = {
    "base": {},
    "bipop": {"local_restart": "BIPOP"},
}

def parse_dat_file(file_path):
    all_runs = []
    current_run = []

    if not os.path.exists(file_path):
        return all_runs

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "evaluations raw_y" in line:
                if current_run:
                    all_runs.append(current_run)
                current_run = []
            else:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        current_run.append((int(parts[0]), float(parts[1])))
                    except Exception:
                        pass

        if current_run:
            all_runs.append(current_run)

    return all_runs

# ============================================================
# 1. Robust problem adapter
# ============================================================

def _to_array_bound(bound, dim, default):
    if bound is None:
        return np.full(dim, default, dtype=float)
    arr = np.asarray(bound, dtype=float)
    if arr.ndim == 0:
        return np.full(dim, float(arr), dtype=float)
    if len(arr) != dim:
        raise ValueError(f"Bound length {len(arr)} != dim {dim}")
    return arr.astype(float)


def get_problem_name(problem, idx):
    for attr in ["name", "problem_name", "__name__"]:
        if hasattr(problem, attr):
            value = getattr(problem, attr)
            if isinstance(value, str) and value:
                return value
    if isinstance(problem, dict):
        for key in ["name", "problem_name"]:
            if key in problem:
                return str(problem[key])
    return f"LLM_{idx:04d}"


def get_problem_dim(problem):
    if hasattr(problem, "meta_data") and hasattr(problem.meta_data, "n_variables"):
        return int(problem.meta_data.n_variables)

    for attr in ["dim", "dimension", "n_variables", "n_var", "n_dims"]:
        if hasattr(problem, attr):
            return int(getattr(problem, attr))

    if isinstance(problem, dict):
        for key in ["dim", "dimension", "n_variables", "n_var", "n_dims"]:
            if key in problem:
                return int(problem[key])
        if "bounds" in problem:
            bounds = problem["bounds"]
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lb = np.asarray(bounds[0])
                if lb.ndim > 0:
                    return len(lb)

    return int(DEFAULT_DIM)


def get_problem_bounds(problem, dim):
    if hasattr(problem, "bounds"):
        b = problem.bounds
        if hasattr(b, "lb") and hasattr(b, "ub"):
            return (
                _to_array_bound(b.lb, dim, DEFAULT_LOWER_BOUND),
                _to_array_bound(b.ub, dim, DEFAULT_UPPER_BOUND),
            )

    lb = None
    ub = None
    for attr in ["lb", "lower_bound", "lower_bounds"]:
        if hasattr(problem, attr):
            lb = getattr(problem, attr)
            break
    for attr in ["ub", "upper_bound", "upper_bounds"]:
        if hasattr(problem, attr):
            ub = getattr(problem, attr)
            break

    if isinstance(problem, dict):
        if "bounds" in problem:
            bounds = problem["bounds"]
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lb, ub = bounds
        lb = problem.get("lb", problem.get("lower_bound", problem.get("lower_bounds", lb)))
        ub = problem.get("ub", problem.get("upper_bound", problem.get("upper_bounds", ub)))

    return (
        _to_array_bound(lb, dim, DEFAULT_LOWER_BOUND),
        _to_array_bound(ub, dim, DEFAULT_UPPER_BOUND),
    )


def call_problem(problem, x):
    if isinstance(problem, dict):
        for key in ["func", "function", "objective", "fitness"]:
            if key in problem and callable(problem[key]):
                return problem[key](x)
        raise ValueError("Dict problem must contain a callable key: func/function/objective/fitness.")
    return problem(x)


class LoggedProblem:
    """
    Minimal IOH-like wrapper for arbitrary LLM-generated callable problems.

    It provides:
      - __call__(x)
      - reset()
      - meta_data.n_variables
      - bounds.lb / bounds.ub

    It also writes a simple .dat file that can be parsed by the existing
    AUC scripts because each run starts with "evaluations raw_y".
    """

    def __init__(self, raw_problem, name, dim, lb, ub, log_file):
        self.raw_problem = raw_problem
        self.name = name
        self.meta_data = SimpleNamespace(n_variables=int(dim), name=name)
        self.bounds = SimpleNamespace(lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float))
        self.log_file = log_file
        self.evaluations = 0
        self._fh = None

    def start_run(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._fh = open(self.log_file, "a")
        self._fh.write("evaluations raw_y\n")
        self._fh.flush()
        self.evaluations = 0

    def close_run(self):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __call__(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x = np.clip(x, self.bounds.lb, self.bounds.ub)
        y = float(call_problem(self.raw_problem, x))

        if not np.isfinite(y):
            y = np.finfo(float).max / 100.0

        self.evaluations += 1
        if self._fh is not None:
            self._fh.write(f"{self.evaluations} {y}\n")

        return y

    def reset(self):
        self.close_run()
        self.evaluations = 0


# ============================================================
# 2. Evaluators
# ============================================================

class Evaluator_Baseline:
    def __init__(self, optimizer):
        self.alg = optimizer

    def __call__(self, func, n_reps):
        for seed in range(n_reps):
            np.random.seed(int(seed))
            budget = int(BUDGET_FACTOR * func.meta_data.n_variables)

            func.start_run()
            try:
                if self.alg.startswith("ng_"):
                    parametrization = ng.p.Array(
                        shape=(func.meta_data.n_variables,)
                    ).set_bounds(func.bounds.lb, func.bounds.ub)
                    optimizer = getattr(ng.optimizers, self.alg[3:])(
                        parametrization=parametrization,
                        budget=budget,
                    )
                    optimizer.minimize(func)

                elif self.alg.startswith("modde_"):
                    params = deepcopy(modde_params[self.alg[6:]])
                    params["lambda_"] *= func.meta_data.n_variables
                    c = ModularDE(
                        func,
                        bound_correction="saturate",
                        budget=budget,
                        **params,
                    )
                    c.run()

                else:
                    params = modcma_params[self.alg[7:]]
                    x0 = np.zeros((func.meta_data.n_variables, 1))
                    c = ModularCMAES(
                        func,
                        d=func.meta_data.n_variables,
                        bound_correction="saturate",
                        budget=budget,
                        x0=x0,
                        **params,
                    )
                    c.run()
            finally:
                func.reset()


class Evaluator_Opytimizer:
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
            print(f"Skipping {self.alg_name}, not found in Opytimizer modules.")
            return

        for seed in range(n_reps):
            np.random.seed(int(seed))
            func.start_run()
            try:
                space = SearchSpace(
                    n_agents=30,
                    n_variables=func.meta_data.n_variables,
                    lower_bound=func.bounds.lb,
                    upper_bound=func.bounds.ub,
                )
                optimizer_instance = opt_class()
                function = Function(helper)
                opt_runner = Opytimizer(space, optimizer_instance, function)
                n_iter = int((func.meta_data.n_variables * BUDGET_FACTOR) / 30)
                opt_runner.start(n_iterations=max(1, n_iter))
            finally:
                func.reset()


class Evaluator_Mealpy:
    def __init__(self, optimizer_name):
        self.alg_name = optimizer_name.replace("mealpy_", "")
        self.models = get_models()

    def __call__(self, func, n_reps):
        term = {"max_fe": func.meta_data.n_variables * BUDGET_FACTOR}
        problem = {
            "fit_func": func,
            "lb": func.bounds.lb,
            "ub": func.bounds.ub,
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        model = next((m for m in self.models if m.name == self.alg_name), None)
        if not model:
            print(f"Skipping {self.alg_name}, not configured in mealpy_helper.py.")
            return

        for seed in range(n_reps):
            np.random.seed(int(seed))
            func.start_run()
            try:
                model_copy = copy(model)
                model_copy.solve(problem, termination=term)
            finally:
                func.reset()


# ============================================================
# 3. Running
# ============================================================

def build_problem_index(problems):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)

    rows = []
    for idx, problem in enumerate(problems):
        problem_id = idx + 1
        problem_name = get_problem_name(problem, problem_id)
        dim = get_problem_dim(problem)
        lb, ub = get_problem_bounds(problem, dim)

        rows.append({
            "problem_type": "LLM",
            "problem_name": problem_name,
            "fid": -200,
            "iid": int(problem_id),
            "llm_problem_id": int(problem_id),
            "dim": int(dim),
            "lower_bound_min": float(np.min(lb)),
            "lower_bound_max": float(np.max(lb)),
            "upper_bound_min": float(np.min(ub)),
            "upper_bound_max": float(np.max(ub)),
        })

    pd.DataFrame(rows).to_csv(INDEX_FILE, index=False)
    print(f"Saved problem index to: {INDEX_FILE}")


def run_unified_optimizer(args):
    alg_key, problem_id = args

    problems = extract_llm_generated_problems()
    raw_problem = problems[int(problem_id) - 1]

    problem_name = get_problem_name(raw_problem, int(problem_id))
    dim = get_problem_dim(raw_problem)
    lb, ub = get_problem_bounds(raw_problem, dim)

    lib_source = TOP_ALGORITHMS[alg_key]
    print(f"Running {alg_key} | LLM iid={problem_id} | {problem_name} | D={dim}")

    if lib_source == "baseline":
        algorithm = Evaluator_Baseline(alg_key)
    elif lib_source == "opytimizer":
        algorithm = Evaluator_Opytimizer(alg_key)
    elif lib_source == "mealpy":
        algorithm = Evaluator_Mealpy(alg_key)
    else:
        return

    folder_path = f"{DATA_FOLDER}/{lib_source.upper()}"
    folder_name = f"{alg_key}_LLM_{int(problem_id)}_D{int(dim)}"
    full_folder = os.path.join(folder_path, folder_name)
    os.makedirs(full_folder, exist_ok=True)

    dat_file = os.path.join(full_folder, "IOHprofiler_fLLM.dat")

    # Avoid appending to stale traces from a previous run.
    if os.path.exists(dat_file):
        existing_runs = parse_dat_file(dat_file)

        if len(existing_runs) >= N_REPS:
            print(f"[Skip] {alg_key} | LLM iid={problem_id} already has {len(existing_runs)} runs.")
            return

        print(f"[Rerun] {alg_key} | LLM iid={problem_id} has only {len(existing_runs)} / {N_REPS} runs.")
        os.remove(dat_file)

    func = LoggedProblem(
        raw_problem=raw_problem,
        name=problem_name,
        dim=dim,
        lb=lb,
        ub=ub,
        log_file=dat_file,
    )

    try:
        algorithm(func, N_REPS)
    except Exception as e:
        print(f"Error running {alg_key} on LLM problem {problem_id}: {e}")
    finally:
        func.reset()


def run_parallel(run_function, arguments):
    arguments = list(arguments)
    if len(arguments) == 0:
        print("No arguments to run.")
        return

    p = Pool(min(MAX_THREADS, len(arguments)))
    p.map(run_function, arguments)
    p.close()
    p.join()


def main():
    problems = extract_llm_generated_problems()
    print(f"Loaded {len(problems)} LLM-generated problems.")

    build_problem_index(problems)

    algnames = list(TOP_ALGORITHMS.keys())
    problem_ids = list(range(1, len(problems) + 1))

    args = [
        (alg, pid)
        for alg in algnames
        for pid in problem_ids
    ]

    print(f"LLM-generated problems: {len(problem_ids)}")
    print(f"Algorithms: {len(algnames)}")
    print(f"Total combinations: {len(args)}")

    run_parallel(run_unified_optimizer, args)


if __name__ == "__main__":
    main()
