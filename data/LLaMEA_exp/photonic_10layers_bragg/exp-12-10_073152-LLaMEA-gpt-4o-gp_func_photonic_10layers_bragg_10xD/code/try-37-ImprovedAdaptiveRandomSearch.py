import numpy as np

class ImprovedAdaptiveRandomSearch:
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
            adaptive_step_size = (ub - lb) / 5  # Initial step size
            stagnation_threshold = 10  # Dynamic threshold for stagnation detection
            adjustment_factor = 1.5  # Factor to adjust stagnation threshold based on performance

            while self.evaluations < self.budget:
                candidate = np.clip(current_best + np.random.uniform(-adaptive_step_size, adaptive_step_size, self.dim), lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1
                
                if candidate_value < current_best_value:
                    current_best = candidate
                    current_best_value = candidate_value
                    no_improvement_count = 0
                    adaptive_step_size *= 1.2  # Increase step size on improvement
                else:
                    no_improvement_count += 1
                    adaptive_step_size *= 0.9  # Decrease step size on no improvement

                if no_improvement_count >= stagnation_threshold:
                    adaptive_step_size *= 0.5  # Significantly reduce step size on stagnation
                    no_improvement_count = 0
                    stagnation_threshold = max(5, int(stagnation_threshold / adjustment_factor))  # Adjust threshold

            if current_best_value < best_value:
                best_solution = current_best
                best_value = current_best_value

            # Multi-start strategy: restart may retain some previous knowledge
            adaptive_step_size = (ub - lb) / (10 + np.random.rand() * 5)  # Randomized reduced step size for restart

        return best_solution