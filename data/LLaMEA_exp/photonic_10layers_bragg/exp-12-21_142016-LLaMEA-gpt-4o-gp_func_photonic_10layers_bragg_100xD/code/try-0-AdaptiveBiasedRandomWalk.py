import numpy as np

class AdaptiveBiasedRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        evaluations = 1

        step_size = 0.1 * (ub - lb)  # Initial step size

        best_position = current_position
        best_value = current_value

        while evaluations < self.budget:
            # Random direction and biased step size
            direction = np.random.uniform(-1.0, 1.0, self.dim)
            direction /= np.linalg.norm(direction)
            
            proposed_position = current_position + step_size * direction
            proposed_position = np.clip(proposed_position, lb, ub)
            
            proposed_value = func(proposed_position)
            evaluations += 1

            if proposed_value < current_value:
                # Move to better position
                current_position = proposed_position
                current_value = proposed_value

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

                # Decrease step size for finer exploration (exploitation)
                step_size *= 0.9
            else:
                # Increase step size for wider exploration
                step_size *= 1.1
            
            # Adaptive mechanism to reset position if stuck
            if evaluations % (self.budget // 10) == 0:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1

        return best_position, best_value