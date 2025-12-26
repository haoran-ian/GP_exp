import numpy as np

class EnhancedAdaptiveBiasedRandomWalk:
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

        success_counter = 0  # Track consecutive successful steps

        while evaluations < self.budget:
            # Random direction with bias for exploration-exploitation
            direction = np.random.normal(size=self.dim)
            direction /= np.linalg.norm(direction)
            
            proposed_position = current_position + step_size * direction
            proposed_position = np.clip(proposed_position, lb, ub)
            
            proposed_value = func(proposed_position)
            evaluations += 1

            if proposed_value < current_value:
                # Move to better position
                current_position = proposed_position
                current_value = proposed_value
                success_counter += 1

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

                # Decrease step size to fine-tune solutions
                step_size *= 0.85
            else:
                # Increase step size for broader exploration
                step_size *= 1.2
                success_counter = 0

            # Adaptive restart if stuck in local optimum
            if success_counter > 10 and evaluations < self.budget:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1
                step_size = 0.1 * (ub - lb)  # Reset step size

            # Dynamic adjustment of step-size based on success rate
            if evaluations % (self.budget // 10) == 0:
                step_size = min(step_size * (1 + success_counter / 10), 0.2 * (ub - lb))

        return best_position, best_value