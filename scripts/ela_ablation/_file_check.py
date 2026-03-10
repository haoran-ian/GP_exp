# fmt: off
import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
from utils.problems_factory import ProblemName
# fmt: on

if __name__ == "__main__":
    problem_ids = list(range(2, 5))
    task_id = int(sys.argv[1])
    if task_id == 0:
        seeds = list(range(100))
        block_coefs = np.linspace(2.0, 5.0, 31)
        n_coefs = np.logspace(np.log10(10), np.log10(1000), 30).astype(int)
        counter_total = 0
        counter_missing = 0
        for problem_id in problem_ids:
            problem_name = ProblemName(problem_id)
            for seed in seeds:
                for block_coef in block_coefs:
                    counter_total += 1
                    file_name = f"{problem_name}-seed:{seed}-block_coef:{block_coef:.1f}-n_coef:{0}.npy"
                    if not os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
                        counter_missing += 1
                        print(f"Missing file: {file_name}")
                for n_coef in n_coefs:
                    counter_total += 1
                    file_name = f"{problem_name}-seed:{seed}-block_coef:{0.0}-n_coef:{n_coef}.npy"
                    if not os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
                        counter_missing += 1
                        print(f"Missing file: {file_name}")
        print(f"Missing {counter_missing} out of {counter_total} files in total.")
