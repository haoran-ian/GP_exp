import numpy as np

class DynamicStepSizeAdaptiveRandomSearch:
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
            max_no_improvement = 3
            adaptive_step_size = (ub - lb) / 4  # Initial larger step size
            reduction_factor = 0.6  # Reduction factor for the step size
            increase_factor = 1.5  # Increase factor for the step size

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= increase_factor  # Increase step size on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8  # Adjust step size reduction on no improvement

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor  # Reduce step size scale
                    no_improvement_count = 0  # Reset count

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy with adaptive adjustment
            adaptive_step_size = (ub - lb) / (4 + self.evaluations / self.budget)  # Restart with dynamically adjusted step size

        return best_solution