import numpy as np

class AdaptiveGradientBasedRandomSearch:
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
            current_solution = np.random.uniform(lb, ub)
            current_value = func(current_solution)
            self.evaluations += 1
            no_improvement_count = 0
            max_no_improvement = 5
            
            adaptive_step_size = (ub - lb) / 5
            reduction_factor = 0.5
            gradient = np.zeros(self.dim)  # Initialize gradient

            while self.evaluations < self.budget:
                # Estimate the gradient by sampling
                perturbation = np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim)
                candidate = np.clip(current_solution + perturbation, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                # Calculate pseudo-gradient
                gradient_estimate = (candidate_value - current_value) / (np.linalg.norm(perturbation) + 1e-8)
                gradient += gradient_estimate * perturbation

                if candidate_value < current_value:
                    current_solution = candidate
                    current_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9

                if no_improvement_count >= max_no_improvement:
                    adaptive_step_size *= reduction_factor
                    no_improvement_count = 0

            # Use gradient to guide next search direction
            if self.evaluations < self.budget:
                gradient_direction = gradient / (np.linalg.norm(gradient) + 1e-8)
                candidate = np.clip(current_solution + adaptive_step_size * gradient_direction, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < best_value:
                    best_solution = candidate
                    best_value = candidate_value

            # Multi-start strategy
            adaptive_step_size = (ub - lb) / 10

        return best_solution