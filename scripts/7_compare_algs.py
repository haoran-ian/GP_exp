# fmt: off
import os
import sys
import ioh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
# fmt: on


def extrat_ys_from_ioh_dat(ioh_dat_path):
    y_runs = []
    f = open(ioh_dat_path, "r")
    lines = f.readlines()
    y = []
    evaluations = 0
    alg_run = 0
    for line in lines:
        if line[:11] == "evaluations":
            y_runs += [y]
            y = []
            evaluations = 0
            alg_run += 1
        else:
            evaluations += 1
            content = line.split(" ")
            y += [[alg_run, evaluations, float(content[1])]]
    f.close()
    return y_runs


def unit_y_runs_from_LLaMEA_exps(y_exps):
    records = []
    keys = ["LLaMEA_run", "alg_run", "evaluations", "raw_y"]
    for i in range(len(y_exps)):
        y_exp = y_exps[i]
        for y_run in y_exp:
            for y in y_run:
                records += [[i] + y]
    df = pd.DataFrame(data=records, columns=keys)
    return df


def calculate_auc_vectorized(group):
    budget = 4500
    filtered = group[
        (group['evaluations'] >= 1) &
        (group['evaluations'] <= budget)
    ].sort_values('evaluations')
    raw_y_positive = filtered['raw_y_normalized'].clip(lower=1e-8)
    auc = (((np.log10(raw_y_positive) + 8) / 8).sum()) / budget
    AOCC = 1 - auc
    group = group.copy()
    group['AOCC'] = AOCC
    return group


def box_plot(df, column: str, by: str, title: str):
    plt.figure(figsize=(8, 5))
    df.boxplot(column=column, by=by, grid=True)
    plt.title('AOCC by Source')
    plt.suptitle(title)
    plt.xlabel('Source')
    plt.ylabel('AOCC')
    plt.tight_layout()
    plt.savefig(f"results/{title}")
    plt.close()


def compare_AOCC_by_source(problem_name: str, algorithm_source_names,
                           algorithm_source_labels,
                           dim: int, nbest: int = 1, LLaMEA_runs: int = 5):
    root_path = f"data/benchmark_algs/{problem_name}"
    y_exps_by_source = []
    dfs = []
    for source_name in algorithm_source_names:
        for i in range(LLaMEA_runs):
            for j in range(nbest):
                exp_folder = os.path.join(
                    root_path, f"{source_name}_run{i}_best{j}")
                if not os.path.exists(exp_folder):
                    continue
                ioh_dat_path = os.path.join(
                    exp_folder,
                    f"data_f60_{problem_name}/IOHprofiler_f60_DIM{dim}.dat")
                y_runs = extrat_ys_from_ioh_dat(ioh_dat_path)
                y_exps_by_source += [y_runs]
        df = unit_y_runs_from_LLaMEA_exps(y_exps_by_source)
        df['source'] = algorithm_source_labels[algorithm_source_names.index(
            source_name)]
        y_exps_by_source = []
        dfs += [df]
    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged['raw_y_normalized'] = (df_merged['raw_y'] - df_merged['raw_y'].min()) / (
        df_merged['raw_y'].max() - df_merged['raw_y'].min())
    df_merged = df_merged.groupby(
        ['LLaMEA_run', 'alg_run', 'source']).apply(calculate_auc_vectorized)
    df_merged = df_merged.reset_index(drop=True)
    selected_columns = ['LLaMEA_run', 'alg_run', 'source', 'AOCC']
    df_AOCC = df_merged[selected_columns].copy()
    df_AOCC = df_AOCC.drop_duplicates(
        subset=['LLaMEA_run', 'alg_run', 'source'])
    df_AOCC = df_AOCC.replace(
        [np.inf, -np.inf], np.nan).dropna(subset=['AOCC'])
    df_AOCC = df_AOCC.sort_values(['AOCC'])
    df_AOCC = df_AOCC.reset_index(drop=True)
    box_plot(df_AOCC, 'AOCC', 'source', 'AOCC_compare_meta_surface_boxplot')


if __name__ == "__main__":
    nbest = 1
    LLaMEA_runs = 5
    dim = 45
    budget_cof = 100
    problem_name = "meta_surface"
    baseline_name = f"BBOB_{dim}D_{budget_cof}xD"
    gp_name = f"gp_func_{problem_name}_{budget_cof}xD"
    compare_AOCC_by_source(problem_name=problem_name,
                           algorithm_source_names=[baseline_name, gp_name],
                           algorithm_source_labels=[
                               'baseline', 'feature-based proxy'],
                           dim=dim, nbest=nbest, LLaMEA_runs=LLaMEA_runs)
