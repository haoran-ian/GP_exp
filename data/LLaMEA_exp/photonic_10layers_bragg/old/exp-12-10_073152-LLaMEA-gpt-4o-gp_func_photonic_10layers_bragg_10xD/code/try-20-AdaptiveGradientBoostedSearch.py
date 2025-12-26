import numpy as np

class AdaptiveGradientBoostedSearch:
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

            while self.evaluations < self.budget:
                # Gradient estimation
                gradient = np.zeros(self.dim)
                epsilon = 1e-5
                for i in range(self.dim):
                    perturb = np.zeros(self.dim)
                    perturb[i] = epsilon
                    forward = np.clip(current_best + perturb, lb, ub)
                    backward = np.clip(current_best - perturb, lb, ub)
                    gradient[i] = (func(forward) - func(backward)) / (2 * epsilon)
                    self.evaluations += 2
                    if self.evaluations >= self.budget:
                        break

                # Combine random exploration with gradient direction
                search_direction = np.random.uniform(-1, 1, self.dim) + gradient
                candidate = np.clip(current_best + adaptive_step_size * search_direction, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.3
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.8

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

                if self.evaluations >= self.budget:
                    break

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            adaptive_step_size = (ub - lb) / 12

        return best_solution