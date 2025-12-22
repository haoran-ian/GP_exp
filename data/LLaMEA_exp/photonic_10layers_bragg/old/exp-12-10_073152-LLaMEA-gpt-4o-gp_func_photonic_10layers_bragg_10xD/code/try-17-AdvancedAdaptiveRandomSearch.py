import numpy as np

class AdvancedAdaptiveRandomSearch:
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
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.5
            learning_rate = 0.1  # New parameter for adaptive learning

            while self.evaluations < self.budget:
                exploration_factor = np.random.randn(self.dim) * adaptive_step_size
                candidate = np.clip(current_best + exploration_factor, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3
                    learning_rate *= 1.05  # Increase learning rate on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8
                    learning_rate *= 0.95  # Decrease learning rate on no improvement
                
                adaptive_step_size = np.clip(adaptive_step_size, (ub - lb) / 100, (ub - lb) / 2)

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 12

        return best_solution