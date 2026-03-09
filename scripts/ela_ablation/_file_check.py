# fmt: off
import os
import sys
sys.path.insert(0, os.getcwd())
from utils.problems_factory import ProblemName
# fmt: on

if __name__ == "__main__":
    problem_ids = list(range(2, 6))
    task_id = int(sys.argv[1])
    if task_id == 0:
        seeds = list(range(100))
        block_coefs = [2, 3, 4, 5]
        n_coefs = [10, 50, 100, 500, 1000, 5000]
        for problem_id in problem_ids:
            problem_name = ProblemName(problem_id)
            for seed in seeds:
                for block_coef in block_coefs:
                    file_name = f"{problem_name}-seed:{seed}-block_coef:{block_coef}-n_coef:{0}.npy"
                    if not os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
                        print(f"Missing file: {file_name}")
                for n_coef in n_coefs:
                    file_name = f"{problem_name}-seed:{seed}-block_coef:{0}-n_coef:{n_coef}.npy"
                    if not os.path.exists(f"data/Ablation_ELA/Y/{file_name}"):
                        print(f"Missing file: {file_name}")
