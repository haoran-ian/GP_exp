"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys
from turtle import down
import ioh
import pandas as pd

sys.path.append(r'/home/ubuntu/GP_Compare/')

# import pytorch_fid
# import cv2
import numpy as np
import torch

from topodiff import dist_util, logger
from topodiff.script_util import create_regressor

from itertools import product
from functools import partial
from pysr import PySRRegressor
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import runParallelFunction, read_pickle
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr
# from gp_fgenerator.compute_ela import 


os.environ['TOPODIFF_LOGDIR'] = './generated'

# current_device_index = th.cuda.current_device()
# current_device_name = th.cuda.get_device_name(current_device_index)
device = 'cuda:0'


class regressor_p_diff_problem():
    def __init__(self, resampling_rate):
        regressor_p_diff_path = './optimization1/model025600.pt'
        logger.log("loading regressor_p_diff...")
        self.nsize = int(64/resampling_rate)
        self.resampling_rate = resampling_rate
        self.regressor_p_diff = create_regressor(regressor_depth=4, in_channels=1,
                                                 image_size=64, regressor_use_fp16=False,
                                                 regressor_width=128,
                                                 regressor_attention_resolutions="32,16,8",
                                                 regressor_use_scale_shift_norm=True,
                                                 regressor_resblock_updown=True,
                                                 regressor_pool="spatial")
        self.regressor_p_diff.load_state_dict(
            dist_util.load_state_dict(
                regressor_p_diff_path, map_location="cuda", weights_only=True)
        )
        self.regressor_p_diff.to(dist_util.dev())
        self.regressor_p_diff.eval()

    def __call__(self, x):
        # x: from -1 to 1, shape in 64x64
        x = x.reshape(self.nsize, self.nsize)
        x = np.repeat(np.repeat(x, self.resampling_rate, axis=0),
                      self.resampling_rate, axis=1)
        x = torch.tensor(x).float().to(device)
        y = self.regressor_p_diff(x.reshape(1, 1, 64, 64),
                                  torch.zeros(1, ).to(device))
        y =  y.tolist()[0][0]
        return y
    


def get_problem(resampling_rate, nsize):
    prob = regressor_p_diff_problem(resampling_rate)
    ioh.problem.wrap_real_problem(prob,
                                  name="regressor_p_diff",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problem = ioh.get_problem("regressor_p_diff", dimension=nsize**2)
    problem.bounds.lb = [-1. for _ in range(nsize**2)]
    problem.bounds.ub = [1. for _ in range(nsize**2)]
    return problem


def get_ela(dim, f, X, y):
    path_base = os.path.join(os.getcwd(), f'ela_regressor_p_diff_{dim}d')
    if not (os.path.isdir(path_base)):
        os.makedirs(path_base)
    df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5,
                           lower_bound=30., upper_bound=250.)
    fid = f.meta_data.problem_id
    filepath = os.path.join(path_base, f'ela_{fid}.csv')
    df_ela.to_csv(filepath, index=False)
# END DEF


# %%
def get_ela_normalize(dim):
    df_ela = pd.read_csv(f"ela_regressor_p_diff_{dim}d/ela_1121.csv")
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
        os.getcwd(), f'ela_regressor_p_diff_{dim}d', 'ela_min.pickle'), dict_min)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_regressor_p_diff_{dim}d', 'ela_max.pickle'), dict_max)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_regressor_p_diff_{dim}d', 'ela_mean.pickle'), dict_mean)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_regressor_p_diff_{dim}d', 'ela_std.pickle'), dict_std)
# END DEF


# %%
def get_ela_corr(dim):
    df_ela = pd.read_csv(f"ela_regressor_p_diff_{dim}d/ela_1121.csv")
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
        os.getcwd(), f'ela_regressor_p_diff_{dim}d', 'ela_corr.pickle'), list_ela)
# END DEF


if __name__ == "__main__":
    resampling_rate = 1  # from 1, 2, 4, 8, 16, 32
    nsize = int(64/resampling_rate)
    problem = get_problem(resampling_rate, nsize)

    dim = problem.meta_data.n_variables
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=[-1.]*dim,
                     upper_bound=[1.]*dim, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    # y = problem(doe_x)
    # np.save(f"regressor_p_diff_resampling_{resampling_rate}.npy", y)
    y = np.load(f"regressor_p_diff_resampling_{resampling_rate}.npy")
    
    # get_ela(dim, problem, doe_x, y)
    # get_ela_corr(dim)
    # get_ela_normalize(dim)
    target_vector = pd.read_csv(f"ela_regressor_p_diff_{dim}d/ela_1121.csv").mean()
    list_ela = read_pickle(f"ela_regressor_p_diff_{dim}d/ela_corr.pickle")
    ela_min = read_pickle(f"ela_regressor_p_diff_{dim}d/ela_min.pickle")
    ela_max = read_pickle(f"ela_regressor_p_diff_{dim}d/ela_max.pickle")
    dist_metric = "cityblock"
    r = 1
    GP_fgen = GP_func_generator(doe_x,
                                target_vector,
                                bs_ratio=0.8,
                                bs_repeat=5,
                                list_ela=list_ela,
                                ela_min=ela_min,
                                ela_max=ela_max,
                                ela_weight={},
                                dist_metric=dist_metric,
                                problem_label=f'regressor_p_diff_{dim}',
                                filepath_save=f'GP_results/regressor_p_diff_{dim}D_{dist_metric}_rep{r}',
                                tree_size=(3, 12),
                                population=50,
                                cxpb=0.5,
                                mutpb=0.1,
                                ngen=50,
                                nhof=1,
                                seed=r,
                                verbose=True)
    hof, pop = GP_fgen()
