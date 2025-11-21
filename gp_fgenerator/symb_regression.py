
import numpy as np
from .compute_ela import bootstrap_ela, diff_vector, dist_wasserstein
    
#%%
def symb_regr(func_, target_vector, bs_ratio, bs_repeat, list_ela, dict_min, dict_max, dict_weight, dist_metric, verbose, points):
    np.seterr(all='ignore')
    penalty = 1e4
    try:
        list_y = []
        for i in range(len(points)):
            y = func_(points[i])
            list_y.append(np.mean(y))
        y = np.array(list_y)
        y[abs(y) < 1e-20] = 0.0
    except:
        return 'syntax_err', penalty
    
    if (np.isnan(y).any() or np.isinf(y).any() or np.var(y)<1e-20):
        return 'y_err', penalty
    
    try:
        candidate_vector = bootstrap_ela(points, y, bs_ratio=bs_ratio, bs_repeat=bs_repeat)
        candidate_vector.replace([np.inf, -np.inf], np.nan, inplace=True)
        # candidate_vector = candidate_vector.mean(axis=0).to_frame().T
    except:
        return 'ela_err', penalty
    
    # fitness = diff_vector(candidate_vector, target_vector, list_ela=list_ela, dict_min=dict_min, dict_max=dict_max, dict_weight=dict_weight, dist_metric=dist_metric)
    fitness = dist_wasserstein(candidate_vector, target_vector, list_ela=list_ela, dict_min=dict_min, dict_max=dict_max, dict_weight=dict_weight)
    if (np.isnan(fitness)):
        return 'dist_err', penalty
    return 'success', min(fitness, penalty), candidate_vector
# END DEF