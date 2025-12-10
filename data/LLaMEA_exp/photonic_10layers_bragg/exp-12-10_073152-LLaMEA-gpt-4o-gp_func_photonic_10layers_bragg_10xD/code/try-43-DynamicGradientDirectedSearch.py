import numpy as np

class DynamicGradientDirectedSearch:
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
            
            # Gradient direction initialization
            grad_direction = np.zeros(self.dim)

            while self.evaluations < self.budget:
                # Gradient approximation
                epsilon = 1e-8
                grad_approx = np.zeros(self.dim)
                for i in range(self.dim):
                    forward = np.copy(current_best)
                    forward[i] += epsilon
                    backward = np.copy(current_best)
                    backward[i] -= epsilon
                    grad_approx[i] = (func(forward) - func(backward)) / (2 * epsilon)
                    self.evaluations += 2  # Two evaluations per dimension

                # Directional move
                grad_direction = 0.9 * grad_direction + 0.1 * grad_approx
                candidate = np.clip(current_best - adaptive_step_size * grad_direction, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 10

        return best_solution