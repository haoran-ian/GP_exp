# fmt: off
import os
import sys
import ioh
import pandas as pd
sys.path.insert(0, os.getcwd())
from problems.meta_surface.problem import get_meta_surface_problem
from problems.photovotaic_problems.problem import PROBLEM_TYPE, get_photonic_problem
from gp_fgenerator.compute_ela import dist_wasserstein
from gp_fgenerator.utils import read_pickle
# fmt: on


def compare_BBOB_real_problem(problem: ioh.problem.RealSingleObjective,
                              problem_name: str):
    dim = problem.meta_data.n_variables
    target = pd.read_csv(f'data/ELA/ela_{problem_name}/ela_60.csv').mean()
    list_ela = read_pickle(f'data/ELA/ela_{problem_name}/ela_corr.pickle')
    ela_min = read_pickle(f'data/ELA/ela_{problem_name}/ela_min.pickle')
    ela_max = read_pickle(f'data/ELA/ela_{problem_name}/ela_max.pickle')
    values = [[]]
    keys = []
    for fid in range(1, 25):
        bbob_ela_path = f'data/ELA/ela_BBOB_f{fid}_d{dim}'
        if not os.path.exists(os.path.join(bbob_ela_path, f'ela_{fid}.csv')):
            continue
        candidate = pd.read_csv(os.path.join(bbob_ela_path, f'ela_{fid}.csv'))
        fitness = dist_wasserstein(candidate_vector=candidate,
                                   target_vector=target, list_ela=list_ela,
                                   dict_min=ela_min, dict_max=ela_max)
        values[0] += [fitness]
        keys += [fid]
    df = pd.DataFrame(data=values, columns=keys)
    df.to_csv(f'data/GP_results/dist_bbob_{problem_name}.csv', index=False)


if __name__ == "__main__":
    problems = [
        get_meta_surface_problem(),
        get_photonic_problem(problem_type=PROBLEM_TYPE.ELLIPSOMETRY),
        get_photonic_problem(num_layers=10, problem_type=PROBLEM_TYPE.BRAGG),
        get_photonic_problem(
            num_layers=10, problem_type=PROBLEM_TYPE.PHOTOVOLTAIC),
        get_photonic_problem(num_layers=20, problem_type=PROBLEM_TYPE.BRAGG),
        get_photonic_problem(
            num_layers=20, problem_type=PROBLEM_TYPE.PHOTOVOLTAIC),
    ]
    problem_names = [
        'meta_surface',
        'photonic_2layers_ellipsometry',
        'photonic_10layers_bragg',
        'photonic_10layers_photovoltaic',
        'photonic_20layers_bragg',
        'photonic_20layers_photovoltaic',
    ]
    for i in range(len(problems)):
        compare_BBOB_real_problem(problems[i], problem_names[i])
