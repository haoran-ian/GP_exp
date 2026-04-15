# fmt: off
import os
import sys
import ioh
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from matplotlib import rcParams
# fmt: on


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


colors = [
    '#1F78B4',
    '#33A02C',
    '#E31A1C',
    '#FF7F00',
    '#6A3D9A',
    '#B15928',
    "#A6CEE3",
]
linestyles = ['solid', 'solid', 'solid', 'solid',
              'dashed', 'dashed', 'dashed',
              'dotted', 'dotted', 'dotted']
rcParams.update({'font.size': 16})


def extrat_ys_from_ioh_dat(ioh_dat_path: str):
    y_runs = []
    f = open(ioh_dat_path, 'r')
    lines = f.readlines()
    y = []
    evaluations = 0
    alg_run = 0
    for line in lines:
        if line[:11] == 'evaluations':
            y_runs += [y]
            y = []
            evaluations = 0
            alg_run += 1
        else:
            evaluations += 1
            content = line.split(' ')
            y += [[alg_run, evaluations, float(content[1])]]
    f.close()
    return y_runs


def unit_y_runs_from_LLaMEA_exps(y_exps):
    records = []
    keys = ['LLaMEA_run', 'alg_run', 'evaluations', 'raw_y']
    for i in range(len(y_exps)):
        y_exp = y_exps[i]
        for y_run in y_exp:
            for y in y_run:
                records += [[i] + y]
    df = pd.DataFrame(data=records, columns=keys)
    return df


def box_plot(df, column: str, by: str, title: str):
    plt.figure(figsize=(8, 5))
    desired_order = ['RS', 'DE', 'LSHADE', 'LLaMEA+BBOB', 'LLaMEA+proxy',
                     'LLaMEA+xgboost', 'LLaMEA+real instance']
    # ax = df.boxplot(column=column, by=by, grid=True, figsize=(12, 8), order=desired_order)
    ax = sns.violinplot(x=by, y=column, data=df,
                        order=desired_order,
                        linewidth=1.5, cut=0, fill=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha='center')
    # plt.title('AOCC by Source')
    # plt.suptitle(title)
    plt.xlabel('Algorithms')
    plt.ylabel('AOCC')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'results/{title}')
    plt.close()


def curve_plot(df, by: str, curve_subset, evaluations: int, title: str):
    plt.figure(figsize=(10, 4.5))
    x = np.arange(evaluations)
    for i in range(len(curve_subset)):
        curve_label = curve_subset[i]
        df_subset = df[(df[by] == curve_label) & (
            df['evaluations'] <= evaluations)]
        window_size = int(evaluations/100)
        smoothed_mean = df_subset['mean'].rolling(window=window_size,
                                                  center=True,
                                                  min_periods=1).median()
        smoothed_std = df_subset['std'].rolling(window=window_size,
                                                center=True,
                                                min_periods=1).median()
        # print(x.shape)
        # print(smoothed_mean.shape)
        # print(curve_label)
        if smoothed_mean.shape[0] < x.shape[0]:
            continue
        plt.plot(x, smoothed_mean, color=colors[i], linestyle=linestyles[i],
                 label=curve_label)
        # plt.fill_between(x, smoothed_mean - smoothed_std,
        #                  smoothed_mean + smoothed_std,
        #                  color=colors[i], alpha=0.05)
    plt.yscale('logit')
    plt.xlabel('evaluations')
    plt.ylabel('fitness')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'results/{title}.png')
    plt.close()


def build_ioh_dat_by_source(problem_name: str, algorithm_source_names,
                            algorithm_source_labels, dim: int, budget_cof: int,
                            nbest: int = 1, LLaMEA_runs: int = 5):
    root_path = f'data/benchmark_algs/{problem_name}'
    y_exps_by_source = []
    dfs = []
    for source_name in algorithm_source_names:
        for i in range(LLaMEA_runs):
            for j in range(nbest):
                exp_folder = os.path.join(
                    root_path, f'{source_name}_run{i}_best{j}')
                if not os.path.exists(exp_folder):
                    print(exp_folder)
                    continue
                ioh_dat_path = os.path.join(
                    exp_folder,
                    f'data_f60_{problem_name}/IOHprofiler_f60_DIM{dim}.dat')
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
    df_merged = df_merged[df_merged['evaluations'] <= budget_cof*dim]
    return df_merged


def select_nbest_algs_from_LLaMEA_runs(df_AOCC, n: int = 1):
    def calculate_mean_AOCC(group):
        group = group.copy()
        group['mean_AOCC'] = group['AOCC'].mean()
        return group
    df = df_AOCC.copy()
    df = df.groupby(['LLaMEA_run', 'source']).apply(calculate_mean_AOCC)
    df = df.reset_index(drop=True)
    selected_columns = ['LLaMEA_run', 'source', 'mean_AOCC']
    df = df[selected_columns]
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return (
        df.groupby('source')
        .apply(lambda x: x.nlargest(n, 'mean_AOCC')['LLaMEA_run'].tolist())
        .to_dict()
    )


def compare_AOCC_by_source(df_merged, problem_name, evaluations: int,
                           nbest: int = 1):
    def calculate_auc_vectorized(group):
        budget = group['evaluations'].max()
        raw_y_positive = group['raw_y_normalized'].clip(lower=1e-8)
        auc = (((np.log10(raw_y_positive) + 8) / 8).sum()) / budget
        AOCC = 1 - auc
        group = group.copy()
        group['AOCC'] = AOCC
        return group
    df_merged = df_merged[df_merged['evaluations'] <= evaluations]
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
    best_LLaMEA_algs = select_nbest_algs_from_LLaMEA_runs(df_AOCC, nbest)
    filtered_rows = []
    for source in df_AOCC['source'].unique():
        if source in best_LLaMEA_algs:
            best_algs = best_LLaMEA_algs[source]
            source_mask = (df_AOCC['source'] == source) & (
                df_AOCC['LLaMEA_run'].isin(best_algs))
            filtered_rows.append(df_AOCC[source_mask])
    df_AOCC_best = pd.concat(filtered_rows, ignore_index=True)
    box_plot(df_AOCC_best, 'AOCC', 'source',
             f'AOCC_compare_{problem_name}_boxplot')
    return best_LLaMEA_algs


def compare_convergence_curve_by_source(df_merged, best_LLaMEA_algs,
                                        problem_name, labels, evaluations):
    def calcuate_std_vectorized(group):
        group = group.copy()
        # group['mean'] = np.mean(group['raw_y_normalized'])
        # group['std'] = np.std(group['raw_y_normalized'])
        group['mean'] = np.mean(group['current_best_y'])
        group['std'] = np.std(group['current_best_y'])
        return group
    df_merged = df_merged.copy()
    conditions = []
    for source, runs in best_LLaMEA_algs.items():
        runs_str = ', '.join(map(str, runs))
        condition = f"(source == '{source}' and LLaMEA_run in [{runs_str}])"
        conditions.append(condition)
    query_str = ' or '.join(conditions)
    df_merged = df_merged.query(query_str)
    df_merged = df_merged.groupby(
        ['LLaMEA_run', 'evaluations', 'source']).apply(calcuate_std_vectorized)
    selected_columns = ['LLaMEA_run', 'evaluations', 'source', 'mean', 'std']
    df = df_merged[selected_columns].copy()
    df = df.drop_duplicates(subset=['LLaMEA_run', 'evaluations', 'source', ])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['mean', 'std'])
    df = df.reset_index(drop=True)
    curve_plot(df, 'source', labels, evaluations,
               f'optimization_curve_compare_{problem_name}')


if __name__ == '__main__':
    nbest = 1
    LLaMEA_runs = 5
    budget_cof = 100
    dims = {'meta_surface': 45,
            # 'meta_surface_solver': 45,
            'photonic_10layers_bragg': 10,
            'photonic_20layers_bragg': 20,
            'photonic_2layers_ellipsometry': 2,
            'photonic_10layers_photovoltaic': 10}
    problem_names = list(dims.keys())
    for i in range(len(problem_names)):
        problem_name = problem_names[i]
        dim = dims[problem_name]
        source_names = [
            f'{problem_name}_{budget_cof}xD',
            f'gp_func_{problem_name}_{budget_cof}xD',
            # f'BBOB_{problem_name}_{budget_cof}xD',
            f'BBOB_{dim}D_{budget_cof}xD' if problem_name not in [
                'photonic_20layers_bragg', 'photonic_10layers_photovoltaic'
            ] else f'BBOB_{problem_name}_{budget_cof}xD',
            f'xgboost_{problem_name}_{budget_cof}xD',
            f'RandomSearch_{budget_cof}xD',
            f'DE_{budget_cof}xD',
            f'LSHADE_{budget_cof}xD',
            # f'CMA-ES_{budget_cof}xD',
        ]
        labels = [
            'LLaMEA+real instance',
            'LLaMEA+proxy',
            'LLaMEA+BBOB',
            'LLaMEA+xgboost',
            'RS',
            'DE',
            'LSHADE',
            # 'CMA-ES',
        ]
        df_merged = build_ioh_dat_by_source(problem_name=problem_name,
                                            algorithm_source_names=source_names,
                                            algorithm_source_labels=labels,
                                            dim=dim, budget_cof=budget_cof,
                                            nbest=nbest, LLaMEA_runs=LLaMEA_runs)
        df_merged['current_best_y'] = (
            df_merged.sort_values(['LLaMEA_run', 'alg_run', 'evaluations'])
            .groupby(['LLaMEA_run', 'alg_run', 'source'])['raw_y_normalized']
            .cummin()
            .reindex(df_merged.index)
        )
        best_LLaMEA_algs = compare_AOCC_by_source(df_merged, problem_name, nbest=1,
                                                  evaluations=dim*50)
        compare_convergence_curve_by_source(df_merged, best_LLaMEA_algs,
                                            problem_name, labels,
                                            evaluations=dim*50)
