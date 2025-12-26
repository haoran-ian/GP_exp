import numpy as np

class AdaptiveDirectionalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        evaluations = 1

        step_size = 0.1 * (ub - lb)  # Initial step size
        directional_memory = np.zeros(self.dim)

        best_position = current_position
        best_value = current_value

        while evaluations < self.budget:
            # Compute new search direction based on past successes
            direction = np.random.uniform(-1.0, 1.0, self.dim) + directional_memory
            direction /= np.linalg.norm(direction)
            
            proposed_position = current_position + step_size * direction
            proposed_position = np.clip(proposed_position, lb, ub)
            
            proposed_value = func(proposed_position)
            evaluations += 1

            if proposed_value < current_value:
                # Successful move; update position and directional memory
                current_position = proposed_position
                current_value = proposed_value
                directional_memory = 0.5 * directional_memory + 0.5 * direction

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

                # Decrease step size for fine-grained exploration
                step_size *= 0.9
            else:
                # Unsuccessful move; increase step size for exploration
                step_size *= 1.1
                directional_memory = 0.9 * directional_memory  # Decay directional memory
            
            # Adaptive restart mechanism to escape local optima
            if evaluations % (self.budget // 20) == 0:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1
                directional_memory = np.zeros(self.dim)

        return best_position, best_value