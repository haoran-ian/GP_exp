import os
import sys
import ioh
import torch
import pandas as pd
import matplotlib.pyplot as plt

from modcma import modularcmaes
from problems.fluid_dynamics.topodiff import dist_util, logger
from problems.fluid_dynamics.topodiff.script_util import create_regressor
from gp_fgenerator.create_pset import *
from gp_fgenerator.sampling import sampling
from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr

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
        y = y.tolist()[0][0]
        return y


def get_topology_problem(resampling_rate, nsize):
    prob = regressor_p_diff_problem(resampling_rate)
    ioh.problem.wrap_real_problem(prob,
                                  name="regressor_p_diff",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problem = ioh.get_problem("regressor_p_diff", dimension=nsize**2)
    problem.bounds.lb = [-1. for _ in range(nsize**2)]
    problem.bounds.ub = [1. for _ in range(nsize**2)]
    return problem


def get_ela(dim, f, X, y, exp_name):
    path_base = os.path.join(os.getcwd(), f"ela_{exp_name}")
    if not (os.path.isdir(path_base)):
        os.makedirs(path_base)
    df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5,
                           lower_bound=30., upper_bound=250.)
    fid = f.meta_data.problem_id
    filepath = os.path.join(path_base, f'ela_{fid}.csv')
    df_ela.to_csv(filepath, index=False)
# END DEF


# %%
def get_ela_normalize(dim, exp_name):
    df_ela = pd.read_csv(f"ela_{exp_name}/ela_1121.csv")
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
        os.getcwd(), f'ela_{exp_name}', 'ela_min.pickle'), dict_min)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_max.pickle'), dict_max)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_mean.pickle'), dict_mean)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_std.pickle'), dict_std)
# END DEF


# %%
def get_ela_corr(dim, exp_name):
    df_ela = pd.read_csv(f"ela_{exp_name}/ela_1121.csv")
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


def test_func(x):
    y = sqrt(sum_vec(add(abs_(neg(ln(add(abs_(sum_vec(add(abs_(neg(ln(add(1.0526382467484297, mul(5.571295299507587, x))))), mul(
        5.571295299507587, x)))), mul(5.571295299507587, x))))), exp(1.0412046398405117))))
    # prob = ioh.get_problem(16, 1, 16, ioh.ProblemClass.BBOB)
    # y = prob(x)
    return y


def ela_process(problem, exp_name):
    dim = problem.meta_data.n_variables
    ndoe = 150*dim
    doe_x = sampling('sobol', n=ndoe, lower_bound=problem.bounds.lb,
                     upper_bound=problem.bounds.ub, round_off=2, random_seed=42,
                     verbose=True).create_doe()
    y = problem(doe_x)
    np.save(f"{exp_name}.npy", y)
    y = np.load(f"{exp_name}.npy")
    get_ela(dim, problem, doe_x, y, exp_name)
    get_ela_corr(dim, exp_name)
    get_ela_normalize(dim, exp_name)


def gp_func_preprocess():
    ndim = 16
    lb = -1.
    ub = 1.
    ioh.problem.wrap_real_problem(test_func,
                                  name="func_gp",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problem = ioh.get_problem("func_gp", dimension=ndim)
    problem.bounds.lb = [lb for _ in range(ndim)]
    problem.bounds.ub = [ub for _ in range(ndim)]
    ela_process(problem, "func_gp_topology_random_16D")


def plot_simple_comparison(df1, df2, df3, df1_name="DF1", df2_name="DF2", df3_name="DF3"):
    keys = df1.columns
    x_positions = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(16, 8))

    for i, (idx, row) in enumerate(df3.iterrows()):
        ax.plot(x_positions, row.values,
                marker='s', linewidth=1, markersize=4,
                color='grey', alpha=0.2,
                label=df3_name if i == 0 else "")

    for i, (idx, row) in enumerate(df1.iterrows()):
        ax.plot(x_positions, row.values,
                marker='o', linewidth=1.5, markersize=4,
                color='red', alpha=0.7,
                label=df1_name if i == 0 else "")

    for i, (idx, row) in enumerate(df2.iterrows()):
        ax.plot(x_positions, row.values,
                marker='s', linewidth=1.5, markersize=4,
                color='blue', alpha=0.7,
                label=df2_name if i == 0 else "")

    for x in x_positions:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Keys')
    ax.set_ylabel('Values')
    ax.set_title('ELA feature compare')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(keys, rotation=90, ha='center', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("ela.png")


def expriment():
    ndim = 16
    lb = -1.
    ub = 1.
    # resampling_rate = 16  # from 1, 2, 4, 8, 16, 32
    # nsize = int(64/resampling_rate)
    ioh.problem.wrap_real_problem(test_func,
                                  name="func_gp",
                                  optimization_type=ioh.OptimizationType.MIN,)
    # origin_problem = get_topology_problem(resampling_rate, nsize)
    # random_problem = ioh.get_problem(16, 1, 16, ioh.ProblemClass.BBOB)
    gp_func = ioh.get_problem("func_gp", dimension=ndim)
    gp_func.bounds.lb = [lb for _ in range(ndim)]
    gp_func.bounds.ub = [ub for _ in range(ndim)]
    problems = [gp_func]
    labels = ["GP Generator"]
    for i in range(1):
        problem = problems[i]
        triggers = [
            ioh.logger.trigger.Each(3),
            ioh.logger.trigger.OnImprovement()
        ]
        ioh_logger = ioh.logger.Analyzer(
            root=os.getcwd(),
            folder_name="problems_ela_compare",
            algorithm_name=f"CMA-ES, {labels[i]}",
            store_positions=True,
            triggers=triggers
        )
        problem.attach_logger(ioh_logger)
        for run in range(10):
            cma = modularcmaes.ModularCMAES(
                problem, ndim, budget=ndim*500).run()
            problem.reset()
        ioh_logger.close()


if __name__ == "__main__":
    # gp_func_preprocess()
    expriment()
    # ela_origin = pd.read_csv("ela_regressor_p_diff_16d/ela_1121.csv")
    # ela_gp_func = pd.read_csv("ela_func_gp_topology_16D/ela_1121.csv")
    # ela_bbob_16 = pd.read_csv("ela_func_gp_topology_random_16D/ela_1121.csv")
    # ela_origin = ela_origin.drop(["ela_meta.lin_simple.coef.max_by_min",
    #                               "ela_meta.quad_simple.cond"], axis=1)
    # ela_gp_func = ela_gp_func.drop(["ela_meta.lin_simple.coef.max_by_min",
    #                                 "ela_meta.quad_simple.cond"], axis=1)
    # ela_bbob_16 = ela_bbob_16.drop(["ela_meta.lin_simple.coef.max_by_min",
    #                                 "ela_meta.quad_simple.cond"], axis=1)
    # if not ela_origin.columns.equals(ela_gp_func.columns):
    #     exit(-1)
    # plot_simple_comparison(ela_origin, ela_gp_func, ela_bbob_16, "topology problem", "GP generator", "BBOB f16")
