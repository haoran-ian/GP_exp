import numpy as np

class AdaptiveGradientAssistedRandomSearch:
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
        step_size = (ub - lb) * 0.1  # Initial step size

        while self.evaluations < self.budget:
            # Calculate gradient approximation
            grad = np.zeros(self.dim)
            epsilon = 1e-8
            for i in range(self.dim):
                perturb = np.zeros(self.dim)
                perturb[i] = epsilon
                grad[i] = (func(current_best + perturb) - current_best_value) / epsilon
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return current_best
            
            # Modulate step size based on improvement
            candidate = np.clip(current_best - step_size * grad + np.random.uniform(-step_size, step_size, self.dim), lb, ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            
            if candidate_value < current_best_value:
                current_best = candidate
                current_best_value = candidate_value
                no_improvement_count = 0
                step_size = min(step_size * 1.2, ub - lb)  # Increase step size on improvement
            else:
                no_improvement_count += 1
                step_size *= 0.5  # Decrease step size
            
            if no_improvement_count >= 15:
                current_best = np.random.uniform(lb, ub)  # Intelligent restart
                current_best_value = func(current_best)
                self.evaluations += 1
                no_improvement_count = 0
                step_size = (ub - lb) * 0.1  # Reset step size

        return current_best