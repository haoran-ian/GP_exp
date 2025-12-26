import numpy as np

class InertiaAdaptiveRestartWalk:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_position = np.random.uniform(lb, ub, self.dim)
        current_value = func(current_position)
        evaluations = 1

        step_size = 0.1 * (ub - lb)
        velocity = np.zeros(self.dim)
        inertia_weight = 0.9
        best_position = current_position
        best_value = current_value
        memory_position = current_position
        memory_value = current_value

        success_count = 0
        adapt_interval = 10
        phase_switch = 0.5
        restart_interval = self.budget // 4

        while evaluations < self.budget:
            if evaluations / self.budget < phase_switch:
                # Phase 1: Exploration with adaptive inertia
                direction = np.random.normal(size=self.dim)
            else:
                # Phase 2: Enhanced Exploitation with inertia and restart
                direction = np.random.uniform(-0.5, 0.5, self.dim)

            direction /= np.linalg.norm(direction)
            velocity = inertia_weight * velocity + step_size * direction
            proposed_position = current_position + velocity
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

                if proposed_value < memory_value:
                    memory_position = proposed_position
                    memory_value = proposed_value

            if evaluations % adapt_interval == 0:
                success_ratio = success_count / adapt_interval
                step_size *= (1 + 0.85 * (success_ratio - 0.5))

                if success_ratio < 0.35:
                    current_position = memory_position
                success_count = 0

            if evaluations % restart_interval == 0:
                random_restart_position = np.random.uniform(lb, ub, self.dim)
                random_restart_value = func(random_restart_position)
                evaluations += 1
                if random_restart_value < memory_value:
                    memory_position = random_restart_position
                    memory_value = random_restart_value

        return best_position, best_value