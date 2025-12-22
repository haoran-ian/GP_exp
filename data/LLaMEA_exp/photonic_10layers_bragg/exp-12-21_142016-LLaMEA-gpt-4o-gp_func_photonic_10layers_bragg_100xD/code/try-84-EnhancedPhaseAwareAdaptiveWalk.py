import numpy as np

class EnhancedPhaseAwareAdaptiveWalk:
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
        phase_memory = {'explore': current_position, 'exploit': current_position}
        phase_memory_value = {'explore': current_value, 'exploit': current_value}

        success_count = 0
        adapt_interval = 10
        phase_switch = 0.5

        while evaluations < self.budget:
            if evaluations / self.budget < phase_switch:
                # Phase 1: Exploration with adaptive reinforcement
                phase = 'explore'
                if success_count < adapt_interval / 2:
                    direction = np.random.normal(size=self.dim)
                else:
                    direction = np.random.uniform(-0.5, 0.5, self.dim)
            else:
                # Phase 2: Exploitation with enhanced reinforcement
                phase = 'exploit'
                if success_count < adapt_interval / 2:
                    direction = np.random.normal(size=self.dim)
                else:
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

                if proposed_value < phase_memory_value[phase]:
                    phase_memory[phase] = proposed_position
                    phase_memory_value[phase] = proposed_value

            if evaluations % adapt_interval == 0:
                success_ratio = success_count / adapt_interval
                step_size *= (1 + 0.85 * (success_ratio - 0.5))

                if success_ratio < 0.35:
                    current_position = phase_memory[phase]
                success_count = 0

            if evaluations % (self.budget // 20) == 0:
                random_checkpoint_position = np.random.uniform(lb, ub, self.dim)
                random_checkpoint_value = func(random_checkpoint_position)
                evaluations += 1
                if random_checkpoint_value < phase_memory_value[phase]:
                    phase_memory[phase] = random_checkpoint_position
                    phase_memory_value[phase] = random_checkpoint_value

        return best_position, best_value