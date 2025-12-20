# fmt: off
import ioh
import numpy as np
import pandas as pd
# fmt: on


def extract_top_bbob(problem_name: str, dim: int, nbest: int = 3):
    instnaces = []
    df = pd.read_csv(f'data/GP_results/dist_bbob_{problem_name}.csv')
    row = df.iloc[0]
    sorted_indices = np.argsort(row.values)[:nbest]
    fids = [int(row.index[i]) for i in sorted_indices]
    for fid in fids:
        instances += [ioh.get_problem(fid=fid, instance=1, dimension=dim,
                                      problem_class=ioh.ProblemClass.BBOB)]
    return instnaces
