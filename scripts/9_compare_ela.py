# fmt: off
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
# fmt: on

nbest = 3
dims = [45, 10, 20, 2, 10]
problem_names = [
    'meta_surface',
    'photonic_10layers_bragg',
    'photonic_20layers_bragg',
    'photonic_2layers_ellipsometry',
    'photonic_10layers_photovoltaic',
]

values = [[None for _ in range(len(dims))] for _ in range(2)]
for i in range(len(dims)):
    dim = dims[i]
    problem_name = problem_names[i]
    gp_path = f'data/GP_results/{problem_name}/gpfg_opt_runs.csv'
    df = pd.read_csv(gp_path)
    df_unique = df.drop_duplicates(subset=['fitness'])
    df_sorted = df_unique.sort_values(by='fitness')
    fitness_list = df_sorted['fitness'].tolist()
    values[0][i] = np.mean(fitness_list[:nbest])
    df = pd.read_csv(f'data/GP_results/dist_bbob_{problem_name}.csv')
    row = df.iloc[0]
    sorted_indices = np.argsort(row.values)[:nbest]
    fids = [int(row.index[i]) for i in sorted_indices]
    values[1][i] = row[fids].mean()
df = pd.DataFrame(data=values, columns=problem_names)
df.to_csv('data/GP_results/func_compare.csv', index=False)
