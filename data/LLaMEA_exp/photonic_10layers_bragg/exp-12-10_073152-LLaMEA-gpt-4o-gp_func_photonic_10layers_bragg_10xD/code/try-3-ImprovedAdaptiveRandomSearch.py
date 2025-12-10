import numpy as np

class ImprovedAdaptiveRandomSearch:
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
        success_count = 0
        adaptation_threshold = 10  # Number of evaluations before adaptation

        while self.evaluations < self.budget:
            step_size = (ub - lb) / (self.evaluations + 1)
            candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1

            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
                success_count += 1
        
            # Adjust step size based on success ratio
            if self.evaluations % adaptation_threshold == 0:
                success_ratio = success_count / adaptation_threshold
                if success_ratio < 0.2:
                    step_size *= 0.9  # Reduce step size
                elif success_ratio > 0.8:
                    step_size *= 1.1  # Increase step size
                success_count = 0  # Reset success count after adaptation

        return current_best