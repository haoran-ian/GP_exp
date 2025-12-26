import numpy as np

class EnhancedQuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 20
        self.max_swarm_size = 50
        self.min_swarm_size = 10
        self.c1 = 2.5
        self.c2 = 2.5
        self.inertia = 0.9
        self.inertia_decrement = 0.99
        self.local_search_prob = 0.1
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_size = self.initial_swarm_size
        positions = np.random.uniform(lb, ub, (swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(p) for p in positions])
        self.evaluations += swarm_size

        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        historical_best_value = global_best_value

        while self.evaluations < self.budget:
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(swarm_size):
                q_factor = np.random.normal(0, 1, self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]) +
                                 q_factor)
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                if np.random.rand() < self.local_search_prob:
                    local_search_step = np.random.normal(0, 0.1, self.dim)
                    positions[i] = np.clip(positions[i] + local_search_step, lb, ub)

                current_value = func(positions[i])
                self.evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_value

            current_best_index = np.argmin(personal_best_values)
            current_best_value = personal_best_values[current_best_index]
            if current_best_value < global_best_value:
                global_best_position = personal_best_positions[current_best_index]
                global_best_value = current_best_value

            if current_best_value < historical_best_value:
                swarm_size = max(swarm_size - 1, self.min_swarm_size)
                positions = positions[:swarm_size]
                velocities = velocities[:swarm_size]
                personal_best_positions = personal_best_positions[:swarm_size]
                personal_best_values = personal_best_values[:swarm_size]

            diversity = np.mean(np.std(positions, axis=0))
            if diversity < (ub - lb).mean() / 20 and swarm_size < self.max_swarm_size:
                additional_positions = np.random.uniform(lb, ub, (1, self.dim))
                additional_velocities = np.zeros((1, self.dim))
                positions = np.vstack((positions, additional_positions))
                velocities = np.vstack((velocities, additional_velocities))
                personal_best_positions = np.vstack((personal_best_positions, additional_positions))
                additional_values = np.array([func(pos) for pos in additional_positions])
                personal_best_values = np.append(personal_best_values, additional_values)
                self.evaluations += 1
                swarm_size += 1

            self.inertia *= self.inertia_decrement
            historical_best_value = global_best_value

        return global_best_position, global_best_value