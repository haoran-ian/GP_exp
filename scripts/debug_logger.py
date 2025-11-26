# fmt: off
import os
import sys
import ioh
import numpy as np
sys.path.insert(0, os.getcwd())
from gp_fgenerator.create_pset import *
from LLaMEA.misc import aoc_logger, correct_aoc, OverBudgetException
# fmt: on


class RandomSearch:
    'Simple random search algorithm'

    def __init__(self, budget: int, dim: int, length: float = 0.0):
        self.n: int = budget
        self.dim = dim
        self.length: float = length

    def __call__(self, problem: ioh.problem.RealSingleObjective) -> None:
        'Evaluate the problem n times with a randomly generated solution'
        for _ in range(self.n):
            # We can use the problems bounds accessor to get information about the problem bounds
            x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
            self.length = np.linalg.norm(x)

            problem(x)


if __name__ == "__main__":
    dim = 23
    budget = 1000*dim
    problem = ioh.get_problem(16, 1, dim, problem_class=ioh.ProblemClass.BBOB)

    # lower = -0.7 if i==0 else -0.6
    l2 = aoc_logger(budget, lower=1e-8, upper=100,
                    triggers=[ioh.logger.trigger.ALWAYS])
    problem.attach_logger(l2)
    try:
        algorithm = RandomSearch(budget=budget, dim=dim)
        algorithm(problem)
    except OverBudgetException:
        pass
    auc = correct_aoc(problem, l2, budget)
    l2.reset(problem)
    problem.reset()
    print(auc)
