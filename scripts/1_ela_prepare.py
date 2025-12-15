# fmt: off
import os
import sys
import ioh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from problems.fluid_dynamics.problem import get_pipes_topology_problem
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.create_pset import *
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr
# fmt: on


def get_ela(f, X, y, exp_name):
    path_base = os.path.join(os.getcwd(), f"data/ELA/ela_{exp_name}")
    if not (os.path.isdir(path_base)):
        os.makedirs(path_base)
    df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5,
                           lower_bound=30., upper_bound=250.)
    fid = f.meta_data.problem_id
    filepath = os.path.join(path_base, f'ela_{fid}.csv')
    df_ela.to_csv(filepath, index=False)


def get_ela_normalize(fid, exp_name):
    df_ela = pd.read_csv(f"data/ELA/ela_{exp_name}/ela_{fid}.csv")
    df_ela = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True,
                          col_allnan=False, col_anynan=False, row_anynan=False,
                          col_null_var=False, row_dupli=False, filter_key=[],
                          reset_index=False, verbose=False)
    dict_min = {}
    dict_max = {}
    dict_mean = {}
    dict_std = {}
    for ela in df_ela.keys():
        dict_min[ela] = df_ela[ela].min(numeric_only=True)
        dict_max[ela] = df_ela[ela].max(numeric_only=True)
        dict_mean[ela] = df_ela[ela].mean(numeric_only=True)
        dict_std[ela] = df_ela[ela].std(numeric_only=True)
    export_pickle(os.path.join(
        os.getcwd(), f'data/ELA/ela_{exp_name}', 'ela_min.pickle'), dict_min)
    export_pickle(os.path.join(
        os.getcwd(), f'data/ELA/ela_{exp_name}', 'ela_max.pickle'), dict_max)
    export_pickle(os.path.join(
        os.getcwd(), f'data/ELA/ela_{exp_name}', 'ela_mean.pickle'), dict_mean)
    export_pickle(os.path.join(
        os.getcwd(), f'data/ELA/ela_{exp_name}', 'ela_std.pickle'), dict_std)


def get_ela_corr(fid, exp_name):
    df_ela = pd.read_csv(f"data/ELA/ela_{exp_name}/ela_{fid}.csv")
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True,
                            col_allnan=True, col_anynan=True, row_anynan=False,
                            col_null_var=True, row_dupli=False, filter_key=[
                                'pca.expl_var.cov_x', 'pca.expl_var.cor_x',
                                'pca.expl_var_PC1.cov_x',
                                'pca.expl_var_PC1.cor_x'],
                            reset_index=False, verbose=True)
    df_corr, df_pair = dropFeatCorr(
        df_clean, corr_thres=0.9, corr_method='pearson', mode='pair',
        ignore_keys=[], verbose=True)
    list_ela = list(df_corr.keys())
    export_pickle(os.path.join(
        os.getcwd(), f"data/ELA/ela_{exp_name}", 'ela_corr.pickle'), list_ela)


if __name__ == "__main__":
    ############################################################################
    # num_pipes = 3
    # iid = 2
    # problem = get_pipes_topology_problem(iid=iid, num_pipes=num_pipes)
    # dim = problem.meta_data.n_variables
    # exp_name = f"topology_{num_pipes}pipes_{dim}D_instance{iid}"
    ############################################################################
    # exp_name = "fluid_dynamics_3pipes_iid0"
    # problem = get_meta_surface_problem()
    # dim = problem.meta_data.n_variables
    ############################################################################
    # problem = get_photonic_problem(20, PROBLEM_TYPE.PHOTOVOLTAIC)
    # dim = problem.meta_data.n_variables
    # exp_name = f"photonic_{dim}layers_photovoltaic"
    ############################################################################
    exp_name = 'meta_surface'
    problem = get_meta_surface_problem()
    dim = problem.meta_data.n_variables
    ############################################################################
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
                     upper_bound=problem.bounds.ub, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    if not os.path.exists(f"data/y/{exp_name}.npy"):
        l1 = ioh.logger.Analyzer(
            folder_name=f"data/ioh_dat/{exp_name}",
            triggers=[ioh.logger.trigger.ALWAYS],
            store_positions=True
        )
        problem.attach_logger(l1)
        y = problem(doe_x)
        np.save(f"data/y/{exp_name}.npy", y)
    y = np.load(f"data/y/{exp_name}.npy")
    fid = problem.meta_data.problem_id
    get_ela(problem, doe_x, y, exp_name)
    get_ela_normalize(fid, exp_name)
    get_ela_corr(fid, exp_name)
