import numpy as np

class EnhancedAdaptiveRandomWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        evaluations = 1

        step_size = 0.1 * (ub - lb) 
        best_position = current_position
        best_value = current_value

        success_count = 0
        adapt_interval = 10  # Interval for adapting step size

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
                success_count += 1

                if proposed_value < best_value:
                    best_position = proposed_position
                    best_value = proposed_value

            if evaluations % adapt_interval == 0:
                success_ratio = success_count / adapt_interval
                step_size *= (1 + (success_ratio - 0.5))
                success_count = 0

            if evaluations % (self.budget // 20) == 0:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1

        return best_position, best_value