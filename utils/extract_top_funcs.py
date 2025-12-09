# fmt: off
import os
import ioh
import pandas as pd
from gp_fgenerator.create_pset import *
from numpy.typing import ArrayLike
# fmt: on


def extract_top_funcs(gp_exp_path: str, dim: int, real_lb: ArrayLike,
                      real_ub: ArrayLike, nbest: int = 1):
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
class cheap_problem_{i}:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
        self.set_y_bias()

    def __call__(self, x):
        x = np.clip(x, self.lb, self.ub)
        return self.cheap_func(x) - self.y_bias

    def set_y_bias(self):
        X = np.random.uniform(self.lb, self.ub, (1000*self.dim, self.dim))
        y = np.array([self.cheap_func(x) for x in X])
        y_min = np.min(y)
        y_max = np.max(y)
        self.y_bias = y_min - (y_max - y_min)

    def cheap_func(self, x):
        y = {expression}
        return np.mean(y)

real_lb = np.array({real_lb.tolist()})
real_ub = np.array({real_ub.tolist()})
cheap_func_{i} = cheap_problem_{i}(real_lb, real_ub)
"""
        exec(func_code, globals())
        cheap_funcs[f'cheap_func_{i}'] = globals()[f'cheap_func_{i}']
        ioh.problem.wrap_real_problem(cheap_funcs[f'cheap_func_{i}'],
                                      name=f"cheap_func_{i}",
                                      optimization_type=ioh.OptimizationType.MIN)
        problem = ioh.get_problem(f"cheap_func_{i}", dimension=dim)
        problem.bounds.lb = real_lb
        problem.bounds.ub = real_ub
        cheap_problems += [problem]
    return cheap_problems
