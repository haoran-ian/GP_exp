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
        improvement_factor = 1.0
        
        while self.evaluations < self.budget:
            step_size = (ub - lb) / (self.evaluations + 1) * improvement_factor
            candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
                improvement_factor = 1.0  # Reset factor on improvement
            else:
                improvement_factor *= 0.95  # Decrease factor if no improvement
        
        return current_best