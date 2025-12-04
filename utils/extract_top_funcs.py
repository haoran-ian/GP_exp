# fmt: off
import os
import ioh
import pandas as pd
from gp_fgenerator.create_pset import *
from numpy.typing import ArrayLike
# fmt: on


def extract_top_funcs(gp_exp_path: str, dim: int, lb: ArrayLike, ub: ArrayLike,
                      nbest: int = 1):
    cheap_problems = []
    cheap_funcs = {}
    df = pd.read_csv(os.path.join(gp_exp_path, "gpfg_opt_runs.csv"))
    df_unique = df.drop_duplicates(subset=["fitness"])
    df_sorted = df_unique.sort_values(by="fitness")
    strf_list = df_sorted["strf"].tolist()
    for i, expression in enumerate(strf_list):
        if i >= nbest:
            break
        func_code = f"""
def cheap_func_{i}(x):
    y = {expression}
    return np.mean(y)
"""
        exec(func_code, globals())
        cheap_funcs[f'cheap_func_{i}'] = globals()[f'cheap_func_{i}']
        ioh.problem.wrap_real_problem(cheap_funcs[f'cheap_func_{i}'],
                                      name=f"cheap_func_{i}",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem(f"cheap_func_{i}", dimension=dim)
        problem.bounds.lb = lb
        problem.bounds.ub = ub
        cheap_problems += [problem]
    return cheap_problems
