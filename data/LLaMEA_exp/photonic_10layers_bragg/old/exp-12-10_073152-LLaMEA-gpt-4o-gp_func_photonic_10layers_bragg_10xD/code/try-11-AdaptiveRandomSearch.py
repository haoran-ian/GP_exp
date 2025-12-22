import numpy as np

class AdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 10
            adaptive_step_size = (ub - lb) / 10  # Initial step size

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.1  # Increase step size on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9  # Decrease step size on no improvement

                if no_improvement_count >= max_no_improvement:
                    break

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

        return best_solution