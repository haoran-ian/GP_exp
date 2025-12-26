import numpy as np

class EnhancedAdaptiveRandomSearchWithHistory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        adaptive_step_size = (ub - lb) / 5  # Initial larger step size
        successful_attempts = 0

        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3  # Increase step size if improvement occurs
                    successful_attempts += 1
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8  # Decrease step size if no improvement

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size /= 1.5  # Further reduce step size to encourage local search
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Dynamic multi-start strategy: adjust step size based on success history
            if successful_attempts > 0:
                adaptive_step_size = (ub - lb) / (12 - min(successful_attempts, 10))
            else:
                adaptive_step_size = (ub - lb) / 10  # Default smaller step size on failure

            successful_attempts = 0  # Reset for next multi-start

        return best_solution