import numpy as np

class ImprovedAdaptiveWalk:
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
        adapt_interval = 10
        phase_switch = [0.3, 0.7]  # Multi-phase: exploration, hybrid, exploitation

        while evaluations < self.budget:
            phase_ratio = evaluations / self.budget

            if phase_ratio < phase_switch[0]:
                direction = np.random.normal(size=self.dim)  # Broad exploration
            elif phase_ratio < phase_switch[1]:
                direction = np.random.uniform(-0.5, 0.5, self.dim)  # Hybrid search
            else:
                direction = np.random.uniform(-1.0, 1.0, self.dim)  # Intensified exploitation

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
                step_size *= (1 + 0.5 * (success_ratio - 0.2))  # Adjusted scaling factor for better stability
                success_count = 0

            if evaluations % (self.budget // 30) == 0:  # More frequent random restart
                random_checkpoint_position = np.random.uniform(lb, ub, self.dim)
                random_checkpoint_value = func(random_checkpoint_position)
                evaluations += 1
                if random_checkpoint_value < best_value:
                    best_position = random_checkpoint_position
                    best_value = random_checkpoint_value

        return best_position, best_value