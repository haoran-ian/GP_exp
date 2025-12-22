import numpy as np

class ChaoticQuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 20
        self.max_swarm_size = 50
        self.min_swarm_size = 10
        self.c1 = 2.0
        self.c2 = 2.0
        self.min_c1 = 0.5
        self.max_c1 = 3.0
        self.min_c2 = 0.5
        self.max_c2 = 3.0
        self.inertia = 0.8
        self.max_inertia = 0.9
        self.min_inertia = 0.4
        self.evaluations = 0
        self.adaptive_q_factor_weight = 0.1
        self.chaos_factor = 0.7

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

        while self.evaluations < self.budget:
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(swarm_size):
                q_factor = np.random.normal(0, self.adaptive_q_factor_weight * np.std(personal_best_positions)/2, self.dim)
                chaotic_r = self.chaos_factor * (1 - r1) * r1  # Logistic map for chaotic behavior
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * chaotic_r * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]) +
                                 q_factor)
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

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

            diversity = np.mean(np.std(positions, axis=0))
            mean_velocity_magnitude = np.mean(np.linalg.norm(velocities, axis=1))

            if mean_velocity_magnitude < (ub - lb).mean() / 15:
                self.inertia = max(self.inertia - 0.05, self.min_inertia)
                self.c1 = min(self.c1 + 0.1, self.max_c1)
                self.c2 = max(self.c2 - 0.05, self.min_c2)
            else:
                self.inertia = min(self.inertia + 0.05, self.max_inertia)
                self.c1 = max(self.c1 - 0.05, self.min_c1)
                self.c2 = min(self.c2 + 0.1, self.max_c2)

            if self.evaluations % (self.budget // 10) == 0:
                self.local_search(func, positions, personal_best_positions, personal_best_values)

            if diversity < (ub - lb).mean() / 25 and swarm_size > self.min_swarm_size:
                swarm_size = max(swarm_size - 1, self.min_swarm_size)
                positions = positions[:swarm_size]
                velocities = velocities[:swarm_size]
                personal_best_positions = personal_best_positions[:swarm_size]
                personal_best_values = personal_best_values[:swarm_size]
            elif diversity > (ub - lb).mean() / 12 and swarm_size < self.max_swarm_size:
                additional_positions = np.random.uniform(lb, ub, (1, self.dim))
                additional_velocities = np.zeros((1, self.dim))
                positions = np.vstack((positions, additional_positions))
                velocities = np.vstack((velocities, additional_velocities))
                personal_best_positions = np.vstack((personal_best_positions, additional_positions))
                additional_values = np.array([func(pos) for pos in additional_positions])
                personal_best_values = np.append(personal_best_values, additional_values)
                self.evaluations += 1
                swarm_size += 1

        return global_best_position, global_best_value
    
    def local_search(self, func, positions, personal_best_positions, personal_best_values):
        for i, pos in enumerate(personal_best_positions):
            perturbation = np.random.normal(0, 0.1, self.dim)
            candidate_position = np.clip(pos + perturbation, func.bounds.lb, func.bounds.ub)
            candidate_value = func(candidate_position)
            self.evaluations += 1
            if candidate_value < personal_best_values[i]:
                personal_best_positions[i] = candidate_position
                personal_best_values[i] = candidate_value