import numpy as np

class EnhancedAdaptiveRandomSearchV2:
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
            reduction_factor = 0.7  # Slightly higher reduction factor for quicker adaptation
            restart_threshold = self.budget // 10  # Restart if budget usage exceeds 10% without improvement

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3  # More aggressive increase on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8  # More aggressive reduction on no improvement

                if no_improvement_count >= max_no_improvement or self.evaluations % restart_threshold == 0:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0
                    # Use a random restart to escape local minima
                    current_best = np.random.uniform(lb, ub)
                    current_best_value = func(current_best)
                    self.evaluations += 1

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 10

        return best_solution