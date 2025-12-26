import numpy as np

class StochasticRefinementGlobalReposition:
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
        memory_position = current_position
        memory_value = current_value

        success_count = 0
        adapt_interval = 10
        phase_switch = 0.5

        while evaluations < self.budget:
            if evaluations / self.budget < phase_switch:
                # Phase 1: Exploration with stochastic refinement
                direction = np.random.normal(size=self.dim) if success_count < adapt_interval / 2 else np.random.uniform(-0.5, 0.5, self.dim)
            else:
                # Phase 2: Exploitation with Memory-Enhanced Search and stochastic perturbation
                perturbation_factor = 0.1 if success_count < adapt_interval / 2 else 0.2
                direction = (np.random.normal(size=self.dim) + perturbation_factor * (memory_position - current_position))
                direction = np.random.uniform(-1.0, 1.0, self.dim) if success_count >= adapt_interval / 2 else direction

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

                if proposed_value < memory_value:
                    memory_position = proposed_position
                    memory_value = proposed_value

            if evaluations % adapt_interval == 0:
                success_ratio = success_count / adapt_interval
                step_size *= (1 + 0.85 * (success_ratio - 0.5))

                if success_ratio < 0.35:
                    current_position = memory_position
                success_count = 0

            if evaluations % (self.budget // 20) == 0:
                # Periodic global reposition to enhance diversity
                global_reposition_position = np.random.uniform(lb, ub, self.dim)
                global_reposition_value = func(global_reposition_position)
                evaluations += 1
                if global_reposition_value < memory_value:
                    memory_position = global_reposition_position
                    memory_value = global_reposition_value

        return best_position, best_value