import numpy as np

class EnhancedRandomSearch:
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

        while self.evaluations < self.budget:
            step_size = (ub - lb) * (0.5 ** (no_improvement_count / 2))
            candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
                no_improvement_count = 0  # Reset on improvement
            else:
                no_improvement_count += 1  # Increment if no improvement

            if no_improvement_count >= 10:
                current_best = np.random.normal(lb, ub)  # Improved stochastic restart
                current_best_value = func(current_best)
                self.evaluations += 1
                no_improvement_count = 0

        return current_best