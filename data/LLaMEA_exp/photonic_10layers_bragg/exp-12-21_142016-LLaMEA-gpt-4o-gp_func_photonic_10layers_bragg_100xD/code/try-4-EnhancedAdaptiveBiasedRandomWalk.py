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
        learning_rate = 0.05  # Initial learning rate

        best_position = current_position
        best_value = current_value
        
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

                step_size *= 1.0 - learning_rate  # Reduce step size
                learning_rate *= 0.95  # Reduce learning rate for stability
            else:
                step_size *= 1.0 + learning_rate  # Increase step size
                learning_rate *= 1.05  # Increase learning rate for exploration
            
            if evaluations % (self.budget // 10) == 0:
                current_position = np.random.uniform(lb, ub, self.dim)
                current_value = func(current_position)
                evaluations += 1

            # Reflective boundary adjustment
            for i in range(self.dim):
                if current_position[i] < lb[i]:
                    current_position[i] = lb[i] + abs(current_position[i] - lb[i])
                elif current_position[i] > ub[i]:
                    current_position[i] = ub[i] - abs(current_position[i] - ub[i])

        return best_position, best_value