# fmt: off
import os
import sys
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from joblib import Parallel, delayed
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
from problems.photovotaic_problems.problem import get_photonic_problem
# fmt: on

warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_features(X, y, problem, blocks, lower_bound=-5.0, upper_bound=5.0,
                       n_jobs=-1, normalize_y=True):
    """
    Use joblib parallel pflacco calculation

    Params:
    - n_jobs: number of jobs, -1 represents using all cpu cores

    Return:
    - df_ela
    """
    def compute_feature(name, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return name, result
        except Exception as e:
            print(f"{name} failed: {e}")
            return name, {}

    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    if (normalize_y):
        y = (y - y.min()) / (y.max() - y.min())
    tasks = [
        ('ela_meta', calculate_ela_meta, X, y),
        ('ela_distr', calculate_ela_distribution, X, y),
        ('ela_level', calculate_ela_level, X, y),
        ('pca', calculate_pca, X, y),
        ('nbc', calculate_nbc, X, y),
        ('disp', calculate_dispersion, X, y),
        ('ic', calculate_information_content, X, y),
        ('ela_conv', calculate_ela_conv, X, y, problem),
        ('ela_conv', calculate_ela_curvate, X, y, problem, dim, lb, ub),
        ('ela_conv', calculate_ela_local, X, y, problem, dim, lb, ub),
        # ('limo', calculate_limo, X, y, lower_bound, upper_bound, blocks),
        ('cm_angle', calculate_cm_angle, X, y, lower_bound, upper_bound, blocks),
        ('cm_grad', calculate_cm_grad, X, y, lower_bound, upper_bound, blocks),
        ('cm_conv', calculate_cm_conv, X, y, lower_bound, upper_bound, blocks),
        ('fitness_distance', calculate_fitness_distance_correlation, X, y),
        ('gradient', calculate_gradient_features, problem, dim, lb, ub),
        ('hill_climbing', calculate_hill_climbing_features, problem, dim, lb, ub),
        ('length_scale', calculate_length_scales_features, problem, dim, lb, ub),
        ('fla_metrics', calculate_sobol_indices_features, problem, dim, lb, ub),
    ]
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
        delayed(compute_feature)(name, func, *args) for name, func, *args in tasks
    )
    ela_ = {}
    results_dict = {}
    for name, result in results:
        results_dict[name] = result
        if isinstance(result, dict):
            ela_.update(result)
        else:
            ela_[name] = result
    df_ela = pd.DataFrame([ela_])
    return df_ela


if __name__ == "__main__":
    feature_id = int(sys.argv[1])
    instance_id = int(sys.argv[2])
    blocks = 3
    feature_set_names = [
        'ela_meta', 'ela_distr', 'ela_level', 'pca', 'nbc', 'disp', 'ic',
        'ela_conv', 'ela_curvate', 'ela_local', 'cm_angle', 'cm_grad', 'cm_conv',
        'fitness_distance', 'gradient', 'hill_climbing', 'length_scale', 'fla_metrics',
    ]
    feature_set = feature_set_names[feature_id]
    if instance_id == 0:
        problem = get_photonic_problem()
    dim = problem.meta_data.n_variables
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    n_sample = dim**blocks * 6
    X = create_initial_sample(dim=dim, n=n_sample, lower_bound=lb,
                              upper_bound=ub, sample_type='lhs', seed=42)
    y = np.array(problem(X.values))
    df_ela = calculate_features(X, y, problem, blocks, lb, ub)
    print(df_ela)
