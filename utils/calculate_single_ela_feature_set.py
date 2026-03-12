# fmt: off
import os
import sys
import ioh
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import calculate_ela_meta
from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.classical_ela_features import calculate_ela_level
from pflacco.classical_ela_features import calculate_dispersion
from pflacco.classical_ela_features import calculate_information_content
from pflacco.classical_ela_features import calculate_nbc
from pflacco.classical_ela_features import calculate_pca
from pflacco.classical_ela_features import calculate_limo
from pflacco.classical_ela_features import calculate_cm_angle
from pflacco.classical_ela_features import calculate_cm_grad
from pflacco.classical_ela_features import calculate_cm_conv
from pflacco.classical_ela_features import calculate_ela_conv
from pflacco.classical_ela_features import calculate_ela_curvate
from pflacco.classical_ela_features import calculate_ela_local
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.misc_features import calculate_gradient_features
from pflacco.misc_features import calculate_hill_climbing_features
from pflacco.misc_features import calculate_length_scales_features
from pflacco.misc_features import calculate_sobol_indices_features
from pflacco.local_optima_network_features import calculate_lon_features
from utils.problems_factory import ProblemName, get_example_problem
# fmt: on

warnings.filterwarnings('ignore', category=RuntimeWarning)

ela_sets = {"ela_meta": 0, "ela_distr": 1, "ela_level": 2,
            "pca": 3, "nbc": 4, "disp": 5, "ic": 6, "ela_conv": 7,
            "ela_curvate": 8, "ela_local": 9, "cm_angle": 10, "cm_grad": 11,
            "cm_conv": 12, "fitness_distance": 13, "gradient": 14,
            "hill_climbing": 15, "length_scale": 16, "fla_metrics": 17}


def calculate_features(X, y, problem: ioh.ProblemClass.REAL, blocks: int,
                       budget_factor: int, ela_set_name: str,
                       normalize_y=True):
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    if (normalize_y):
        y = (y - y.min()) / (y.max() - y.min())
    if ela_set_name == "ela_meta":
        results = calculate_ela_meta(X, y)
    elif ela_set_name == "ela_distr":
        results = calculate_ela_distribution(X, y)
    elif ela_set_name == "ela_level":
        results = calculate_ela_level(X, y)
    elif ela_set_name == "pca":
        results = calculate_pca(X, y)
    elif ela_set_name == "nbc":
        results = calculate_nbc(X, y)
    elif ela_set_name == "disp":
        results = calculate_dispersion(X, y)
    elif ela_set_name == "ic":
        results = calculate_information_content(X, y)
    elif ela_set_name == "ela_conv":
        results = calculate_ela_conv(X, y, problem)
    elif ela_set_name == "ela_curvate":
        results = calculate_ela_curvate(X, y, problem, dim, lb, ub)
    elif ela_set_name == "ela_local":
        results = calculate_ela_local(X, y, problem, dim, lb, ub)
    elif ela_set_name == "cm_angle":
        results = calculate_cm_angle(X, y, lb, ub, blocks)
    elif ela_set_name == "cm_grad":
        results = calculate_cm_grad(X, y, lb, ub, blocks)
    elif ela_set_name == "cm_conv":
        results = calculate_cm_conv(X, y, lb, ub, blocks)
    elif ela_set_name == "fitness_distance":
        results = calculate_fitness_distance_correlation(X, y)
    elif ela_set_name == "gradient":
        results = calculate_gradient_features(
            problem, dim, lb, ub, budget_factor_per_dim=budget_factor)
    elif ela_set_name == "hill_climbing":
        results = calculate_hill_climbing_features(
            problem, dim, lb, ub, budget_factor_per_run=budget_factor)
    elif ela_set_name == "length_scale":
        results = calculate_length_scales_features(
            problem, dim, lb, ub, budget_factor_per_dim=budget_factor)
    elif ela_set_name == "fla_metrics":
        results = calculate_sobol_indices_features(
            problem, dim, lb, ub, sampling_coefficient=budget_factor)
    df = pd.DataFrame([results])
    df = df.loc[:, ~df.columns.str.contains("costs_runtime")]
    return df


if __name__ == "__main__":
    blocks = 3
################################################################################
    problem_name = ProblemName(int(sys.argv[1]))
    ela_set_name = list(ela_sets.keys())[list(
        ela_sets.values()).index(int(sys.argv[2]))]
    seed = int(sys.argv[3])
    block_coef = float(sys.argv[4])
    n_coef = int(sys.argv[5])
################################################################################
    problem = get_example_problem(problem_name)
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    file_name = f"{problem_name}-seed:{seed}-block_coef:{block_coef:.1f}-n_coef:{n_coef}.npy"
    X = np.load(f"data/Ablation_ELA/X/{file_name}")
    y = np.load(f"data/Ablation_ELA/Y/{file_name}")
    df = calculate_features(X, y, problem, blocks, n_coef, ela_set_name)
    df.to_csv(
        f"data/Ablation_ELA/atom/{problem_name}-{ela_sets[ela_set_name]}-seed:{seed}-block_coef:{block_coef}-n_coef:{n_coef}.csv")
    print(df)
