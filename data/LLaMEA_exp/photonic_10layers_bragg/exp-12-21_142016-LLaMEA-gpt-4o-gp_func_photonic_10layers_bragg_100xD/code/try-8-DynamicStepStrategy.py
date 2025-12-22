import numpy as np

class DynamicStepStrategy:
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

        stagnation_counter = 0
        stagnation_limit = self.budget // 10  # Trigger diversification after a certain number of stagnant evaluations

        while evaluations < self.budget:
            direction = np.random.uniform(-1.0, 1.0, self.dim)
            direction /= np.linalg.norm(direction)

            proposed_position = current_position + step_size * direction
            proposed_position = np.clip(proposed_position, lb, ub)

            proposed_value = func(proposed_position)
            evaluations += 1

            if proposed_value < current_value:
                current_position = proposed_position
                current_value = proposed_value

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

                step_size *= 0.9
                stagnation_counter = 0  # Reset stagnation counter
            else:
                step_size *= 1.1  # Increase more aggressively for better exploration
                stagnation_counter += 1

            if stagnation_counter > stagnation_limit:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1
                stagnation_counter = 0  # Reset after diversification

        return best_position, best_value