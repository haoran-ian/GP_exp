import numpy as np

class DynamicStepSizeRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        evaluations = 1

        step_size = 0.1 * (ub - lb)  # Initial step size
        momentum = np.zeros(self.dim)  # Momentum for direction persistence

        best_position = current_position
        best_value = current_value

        while evaluations < self.budget:
            # Random direction with momentum influence
            direction = np.random.uniform(-1.0, 1.0, self.dim)
            direction = 0.8 * momentum + 0.2 * direction
            direction /= np.linalg.norm(direction)

            proposed_position = current_position + step_size * direction
            proposed_position = np.clip(proposed_position, lb, ub)

            proposed_value = func(proposed_position)
            evaluations += 1

            if proposed_value < current_value:
                # Move to better position
                current_position = proposed_position
                current_value = proposed_value
                momentum = direction  # Update momentum

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

                # Decrease step size for finer exploration (exploitation)
                step_size *= 0.9
            else:
                # Increase step size for wider exploration
                step_size *= 1.1
                momentum = np.zeros(self.dim)  # Reset momentum on failure

            # Adaptive mechanism to reset position if diversity is low
            if evaluations % (self.budget // 10) == 0:
                diversity_check = np.std(current_position)
                if diversity_check < 0.01 * (ub - lb).mean():
                    current_position = np.random.uniform(lb, ub, self.dim)
                    current_value = func(current_position)
                    evaluations += 1
                    step_size = 0.1 * (ub - lb)  # Reset step size
                    momentum = np.zeros(self.dim)  # Reset momentum

        return best_position, best_value