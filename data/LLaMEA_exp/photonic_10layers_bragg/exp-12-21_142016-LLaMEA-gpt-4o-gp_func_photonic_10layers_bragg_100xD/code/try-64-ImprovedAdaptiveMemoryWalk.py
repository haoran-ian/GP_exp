import numpy as np

class ImprovedAdaptiveMemoryWalk:
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
        memory_positions = [current_position]
        memory_values = [current_value]
        
        success_count = 0
        adapt_interval = 10
        phase_switch = 0.5
        memory_rate = 0.1

        while evaluations < self.budget:
            if evaluations / self.budget < phase_switch:
                if success_count < adapt_interval / 2:
                    direction = np.random.normal(size=self.dim)
                else:
                    direction = np.random.uniform(-0.5, 0.5, self.dim)
            else:
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

                if proposed_value < min(memory_values):
                    memory_positions.append(proposed_position)
                    memory_values.append(proposed_value)

            if evaluations % adapt_interval == 0:
                success_ratio = success_count / adapt_interval
                step_size *= (1 + 0.85 * (success_ratio - 0.5))

                if success_ratio < 0.35 and memory_positions:
                    current_position = memory_positions[np.argmin(memory_values)]
                    memory_positions = memory_positions[-int(memory_rate * len(memory_positions)):]
                    memory_values = memory_values[-int(memory_rate * len(memory_values)):]
                success_count = 0

            if evaluations % (self.budget // 20) == 0:
                random_checkpoint_position = np.random.uniform(lb, ub, self.dim)
                random_checkpoint_value = func(random_checkpoint_position)
                evaluations += 1
                if random_checkpoint_value < min(memory_values):
                    memory_positions.append(random_checkpoint_position)
                    memory_values.append(random_checkpoint_value)

        return best_position, best_value