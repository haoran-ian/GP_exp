# fmt: off
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from utils.problems_factory import ProblemName
from utils.calculate_single_ela_feature_set import ela_sets
# fmt: on


def merge_dfs(dfs):
    reference_keys = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], start=1):
        current_keys = set(df.columns)
        if current_keys != reference_keys:
            missing = reference_keys - current_keys
            extra = current_keys - reference_keys
            raise ValueError(
                f"DF {i} keys not match! \nMissing: {missing}\nExtra: {extra}")
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    return combined_df


def build_atom_data_seed(problem_name, ela_name, block_coef, n_coef):
    counter_total = 0
    counter_missing = 0
    dfs = []
    for seed in range(100):
        counter_total += 1
        file_name = f"{problem_name}-{ela_sets[ela_name]}-seed:{seed}-block_coef:{block_coef}-n_coef:{n_coef}.csv"
        if not os.path.exists(f"data/Ablation_ELA/atom/{file_name}"):
            counter_missing += 1
            continue
        df_atom = pd.read_csv(
            f"data/Ablation_ELA/atom/{file_name}", index_col=0)
        df_atom.insert(0, "block_coef", block_coef)
        df_atom.insert(1, "n_coef", n_coef)
        df_atom.insert(2, "seed", seed)
        dfs += [df_atom]
    if len(dfs) == 0:
        return pd.DataFrame()
    df = merge_dfs(dfs)
    print(f"{problem_name} with {ela_name}, block_coef:{block_coef}, n_coef:{n_coef} missing {counter_missing} files out of {counter_total}")
    return df


if __name__ == "__main__":
    for pid in range(2, 4):
        problem_name = ProblemName(pid)
        if not os.path.exists(f"results/ela_convergence/{problem_name}/"):
            os.mkdir(f"results/ela_convergence/{problem_name}/")
        if not os.path.exists(f"results/ela_convergence/{problem_name}/aggregation/"):
            os.mkdir(f"results/ela_convergence/{problem_name}/aggregation/")
        plot_dfs = []
        for ela_set_id in range(7):
            ela_name = list(ela_sets.keys())[list(
                ela_sets.values()).index(ela_set_id)]
            if "cm" in ela_name:
                pass
            else:
                block_coef = 0.0
                dfs = []
                for n_coef in np.logspace(np.log10(10), np.log10(1000), 30).astype(int):
                    df = build_atom_data_seed(
                        problem_name, ela_name, block_coef, n_coef)
                    if not df.empty:
                        dfs += [df]
                df = merge_dfs(dfs)
                group_col = "n_coef"
                target_cols = df.columns[3:]
                agg_df = df.groupby("n_coef")[target_cols].agg(["mean", "std"])
                target_cols = [c for c in agg_df.columns.get_level_values(0).unique() if c != 'n_coef']
                for col in target_cols:
                    std_series = agg_df[col]['std']
                    std_min = std_series.min()
                    std_max = std_series.max()
                    if std_max - std_min != 0:
                        norm_values = (std_series - std_min) / (std_max - std_min)
                    else:
                        norm_values = 0.0
                    agg_df[(col, 'std_norm')] = norm_values
                agg_df = agg_df.sort_index(axis=1)
                norm_data = agg_df.xs('std_norm', axis=1, level=1)
                plot_df = norm_data.stack().reset_index()
                plot_df.columns = ['n_coef', 'feature_name', 'std_norm']
                plot_dfs += [plot_df]
                for feature in list(df.keys())[3:]:
                    sns.lineplot(data=agg_df, x="n_coef", y=(feature, "std"))
                    plt.xscale("log")
                    plt.ylabel("std")
                    plt.title(feature)
                    plt.tight_layout()
                    plt.savefig(
                        f"results/ela_convergence/{problem_name}/{feature}.png")
                    plt.close()
                sns.lineplot(data=plot_df, x="n_coef", y="std_norm")
                plt.xscale("log")
                plt.ylabel("std")
                plt.title(ela_name)
                plt.tight_layout()
                plt.savefig(
                    f"results/ela_convergence/{problem_name}/aggregation/{ela_name}.png")
                plt.close()
        df = merge_dfs(plot_dfs)
        sns.lineplot(data=df, x="n_coef", y="std_norm")
        plt.xscale("log")
        plt.ylabel("std")
        plt.title(problem_name)
        plt.tight_layout()
        plt.savefig(
            f"results/ela_convergence/{problem_name}/aggregation/{problem_name}.png")
        plt.close()
                
