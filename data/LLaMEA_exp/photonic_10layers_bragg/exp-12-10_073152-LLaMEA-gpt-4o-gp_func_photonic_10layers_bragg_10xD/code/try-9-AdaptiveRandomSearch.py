import numpy as np

class AdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        current_best = np.random.uniform(lb, ub)
        current_best_value = func(current_best)
        self.evaluations += 1
        no_improvement_count = 0
        max_stagnation = 10

        while self.evaluations < self.budget:
            adaptive_factor = 1 + 0.1 * (self.budget - self.evaluations) / self.budget
            step_size = (ub - lb) * (0.5 ** no_improvement_count) * adaptive_factor
            candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= max_stagnation:
                current_best = np.random.uniform(lb, ub)
                current_best_value = func(current_best)
                self.evaluations += 1
                max_stagnation = int(max_stagnation * 0.9)
                no_improvement_count = 0

        return current_best