import numpy as np

class EnhancedAdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_solution = None
        best_value = float('inf')
        convergence_rate_threshold = 0.01

        while self.evaluations < self.budget:
            current_best = np.random.uniform(lb, ub)
            current_best_value = func(current_best)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            adaptive_step_size = (ub - lb) / 4  # Initial larger step size
            reduction_factor = 0.5
            change_momentum = 1.1

            last_best_value = current_best_value

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= change_momentum  # Increase step size

                else:
                    no_improvement_count += 1
                
                # Adapt step size based on no improvement and convergence rate
                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor  # Reduce step size scale
                    no_improvement_count = 0  # Reset count
                
                # Dynamically adjust based on convergence rate
                convergence_rate = abs((last_best_value - current_best_value) / last_best_value)
                if convergence_rate < convergence_rate_threshold:
                    adaptive_step_size *= reduction_factor  # Slow convergence, refine search space
                else:
                    adaptive_step_size *= change_momentum  # Fast convergence, expand search space

                last_best_value = current_best_value

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy with diversity preservation
            adaptive_step_size = (ub - lb) / 8  # Restart with a refined step size

        return best_solution