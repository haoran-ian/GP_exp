import numpy as np

class DualPhaseAdaptiveRandomSearch:
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

        phase_transition = self.budget // 2
        
        while self.evaluations < self.budget:
            if self.evaluations < phase_transition:
                # Global exploration phase
                step_size = (ub - lb) * (1 - self.evaluations / self.budget)
            else:
                # Local exploitation phase
                step_size = (ub - lb) / (self.evaluations - phase_transition + 1)

            candidate = np.clip(current_best + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
        
        return current_best