import math
import statistics
import numpy as np
import pandas as pd
import pflacco.classical_ela_features as pflacco_ela
from scipy import spatial
from scipy.stats import wasserstein_distance
from sklearn.utils import resample
from joblib import Parallel, delayed
from gp_fgenerator.utils import dataCleaning


def calculate_features_parallel_joblib(X, y, lower_bound=-5.0, upper_bound=5.0,
                                       n_jobs=-1):
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

    tasks = [
        ('ela_meta', pflacco_ela.calculate_ela_meta, X, y),
        ('ela_distr', pflacco_ela.calculate_ela_distribution, X, y),
        ('ela_level', pflacco_ela.calculate_ela_level, X, y),
        ('pca', pflacco_ela.calculate_pca, X, y),
        # ('limo', pflacco_ela.calculate_limo, X, y, lower_bound, upper_bound),
        ('nbc', pflacco_ela.calculate_nbc, X, y),
        ('disp', pflacco_ela.calculate_dispersion, X, y),
        ('ic', pflacco_ela.calculate_information_content, X, y)
    ]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
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


def compute_ela(X, y, lower_bound=-5.0, upper_bound=5.0, normalize_y=True):
    if (normalize_y):
        y = (y - y.min()) / (y.max() - y.min())
    df_ela = calculate_features_parallel_joblib(X, y, lower_bound, upper_bound)
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=False, col_allnan=False, col_anynan=False, row_anynan=False, col_null_var=False,
                            row_dupli=False, filter_key=['.costs_runtime'], reset_index=False, verbose=False)
    return df_clean
# END DEF

# %%


def bootstrap_ela(X, y, lower_bound=-5.0, upper_bound=5.0, normalize_y=True,
                  bs_ratio=0.8, bs_repeat=2, bs_seed=42, n_jobs=-1):
    assert 0.0 < bs_ratio < 1.0
    assert bs_repeat > 0
    num_sample = int(math.ceil(len(X) * bs_ratio))
    df_ela = pd.DataFrame()
    def process_one_iteration(i_bs, X, y, lower_bound, upper_bound,
                              normalize_y, num_sample, bs_seed):
        i_bs_seed = i_bs + bs_seed
        X_, y_ = resample(X, y, replace=False, n_samples=num_sample,
                          random_state=i_bs_seed, stratify=None)
        ela_ = compute_ela(X_, y_, lower_bound=lower_bound,
                           upper_bound=upper_bound, normalize_y=normalize_y)
        return ela_
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_one_iteration)(
            i_bs, X, y, lower_bound, upper_bound,
            normalize_y, num_sample, bs_seed
        )
        for i_bs in range(bs_repeat)
    )
    df_ela = pd.concat(results, axis=0, ignore_index=True)
    return df_ela
# END DEF

# %%


def process_ela(candidate_ela, target_ela, list_ela=[], dict_min={}, dict_max={}, dict_weight={}):
    if (list_ela):
        candidate_ela = candidate_ela[list_ela]
        target_ela = target_ela[list_ela]
    if (dict_min and dict_max):
        list_min = [dict_min[ela] for ela in candidate_ela.keys()]
        list_diff = [dict_max[ela]-dict_min[ela]
                     for ela in candidate_ela.keys()]
        candidate_ela = (candidate_ela-list_min)/list_diff
        target_ela = (target_ela-list_min)/list_diff
    if (dict_weight):
        list_weight = [dict_weight[ela] for ela in candidate_ela.keys()]
        candidate_ela = candidate_ela*list_weight
        target_ela = target_ela*list_weight
    return candidate_ela, target_ela
# END DEF

# %%


def diff_vector(candidate_vector, target_vector, list_ela=[], dict_min={}, dict_max={}, dict_weight={}, dist_metric='cityblock'):
    assert len(candidate_vector) == len(target_vector)
    dict_dist_metric = {'canberra': spatial.distance.canberra,
                        'cosine': spatial.distance.cosine,
                        'correlation': spatial.distance.correlation,
                        'euclidean': spatial.distance.euclidean,
                        'cityblock': spatial.distance.cityblock}
    cand_, targ_ = process_ela(candidate_vector, target_vector, list_ela=list_ela,
                               dict_min=dict_min, dict_max=dict_max, dict_weight=dict_weight)
    list_dist = []
    for i in range(len(cand_)):
        dist_ = dict_dist_metric[dist_metric](
            np.array(cand_.iloc[i]), np.array(targ_.iloc[i]))
        list_dist.append(dist_)
    return statistics.fmean(list_dist)
# END DEF

# %%


def dist_wasserstein(candidate_vector, target_vector, list_ela=[], dict_min={}, dict_max={}, dict_weight={}):
    cand_, targ_ = process_ela(candidate_vector, target_vector, list_ela=list_ela,
                               dict_min=dict_min, dict_max=dict_max, dict_weight=dict_weight)
    list_dist = []
    cand_values = cand_.values
    targ_values = targ_.values
    combined = np.vstack([cand_values, targ_values])
    col_mins = np.min(combined, axis=0, keepdims=True)
    col_maxs = np.max(combined, axis=0, keepdims=True)
    col_ranges = col_maxs - col_mins
    col_ranges[col_ranges == 0] = 1
    cand_values = (cand_values - col_mins) / col_ranges
    targ_values = (targ_values - col_mins) / col_ranges
    for i in range(cand_.shape[1]):
        list_dist.append(wasserstein_distance(
            list(cand_values[:, i]), [targ_values[i]]))
    # for ela in cand_.keys():
    #     list_dist.append(wasserstein_distance(list(cand_[ela]), list(targ_[ela])))
    return statistics.fmean(list_dist)
# END DEF
