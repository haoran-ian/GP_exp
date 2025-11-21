import os
import re
import sys
import ioh
import time
import numpy as np
import pandas as pd
from photonics_benchmark import *
from itertools import product
from functools import partial
from pysr import PySRRegressor
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import runParallelFunction, read_pickle
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr


def get_photonic_instances(problem_name):
    problems = []
    if problem_name == "bragg":
        # ------- define "mini-bragg" optimization problem
        nb_layers = 10     # number of layers of full stack
        target_wl = 600.0  # nm
        mat_env = 1.0      # materials: ref. index
        mat1 = 1.4
        mat2 = 1.8
        prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
        ioh.problem.wrap_real_problem(prob, name="brag_mirror",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("brag_mirror", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    elif problem_name == "ellipsometry":
        # ------- define "ellipsometry" optimization problem
        mat_env = 1.0
        mat_substrate = 'Gold'
        nb_layers = 1
        min_thick = 50     # nm
        max_thick = 150
        min_eps = 1.1      # permittivity
        max_eps = 3
        wavelengths = np.linspace(400, 800, 100)  # nm
        angle = 40*np.pi/180  # rad
        prob = ellipsometry(mat_env, mat_substrate, nb_layers, min_thick, max_thick,
                            min_eps, max_eps, wavelengths, angle)
        ioh.problem.wrap_real_problem(prob, name="ellipsometry",
                                      optimization_type=ioh.OptimizationType.MIN,)
        problem = ioh.get_problem("ellipsometry", dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    elif problem_name == "photovoltaic":
        # ------- define "sophisticated antireflection" optimization problem
        nb_layers = 30
        min_thick = 30
        max_thick = 250
        wl_min = 375
        wl_max = 750
        prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                   wl_min, wl_max)
        ioh.problem.wrap_real_problem(prob, name="sophisticated_antireflection_design",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem("sophisticated_antireflection_design",
                                  dimension=prob.n)
        problem.bounds.lb = prob.lb
        problem.bounds.ub = prob.ub
        problems.append(problem)
    return problems[0]

# %%


def get_ela(dim, f, X, y):
    path_base = os.path.join(os.getcwd(), f'results_ela_{dim}d')
    if not (os.path.isdir(path_base)):
        os.makedirs(path_base)
    df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5,
                           lower_bound=30., upper_bound=250.)
    fid = f.meta_data.problem_id
    filepath = os.path.join(path_base, f'ela_bbob_{fid}.csv')
    df_ela.to_csv(filepath, index=False)
# END DEF


# %%
def get_ela_normalize(dim):
    df_ela = pd.read_csv("results_ela_30d/ela_bbob_1121.csv")
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
        os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_min.pickle'), dict_min)
    export_pickle(os.path.join(
        os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_max.pickle'), dict_max)
    export_pickle(os.path.join(
        os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_mean.pickle'), dict_mean)
    export_pickle(os.path.join(
        os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_std.pickle'), dict_std)
# END DEF


# %%
def get_ela_corr(dim):
    df_ela = pd.read_csv("results_ela_30d/ela_bbob_1121.csv")
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
        os.getcwd(), f'results_ela_{dim}d', 'ela_bbob_corr.pickle'), list_ela)
# END DEF


if __name__ == "__main__":
    problem_id = 2
    problem_types = ["bragg", "ellipsometry", "photovoltaic"]
    problem_name = problem_types[problem_id]
    problem = get_photonic_instances(problem_name)

    dim = problem.meta_data.n_variables
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=[30.]*dim,
                     upper_bound=[250.]*dim, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    # y = problem(doe_x)
    # np.save("phtovotaic_30D.npy", y)
    y = np.load("phtovotaic_30D.npy")

    # get_ela(dim, problem, doe_x, y)
    # get_ela_corr(dim)
    # get_ela_normalize(dim)
    # target_vector = pd.read_csv("results_ela_30d/ela_bbob_1121.csv").mean()
    # list_ela = read_pickle("results_ela_30d/ela_bbob_corr.pickle")
    # ela_min = read_pickle("results_ela_30d/ela_bbob_min.pickle")
    # ela_max = read_pickle("results_ela_30d/ela_bbob_max.pickle")
    # dist_metric = "cityblock"
    # r = 1
    GP_fgen = GP_func_generator(doe_x,
                                target_vector,
                                bs_ratio=0.8,
                                bs_repeat=5,
                                list_ela=list_ela,
                                ela_min=ela_min,
                                ela_max=ela_max,
                                ela_weight={},
                                dist_metric=dist_metric,
                                problem_label=f'photovotaic_{dim}',
                                filepath_save=f'GP_results/{dim}D_{dist_metric}_rep{r}',
                                tree_size=(3, 12),
                                population=50,
                                cxpb=0.5,
                                mutpb=0.1,
                                ngen=50,
                                nhof=1,
                                seed=r,
                                verbose=True)
    # hof, pop = GP_fgen()

    model = PySRRegressor(
        maxsize=1000,
        niterations=100,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(doe_x, y*10000)
    print(model)
