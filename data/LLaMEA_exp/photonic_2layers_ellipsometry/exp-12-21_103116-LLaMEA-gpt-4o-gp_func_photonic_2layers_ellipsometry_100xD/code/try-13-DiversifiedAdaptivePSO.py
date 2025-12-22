import numpy as np

class DiversifiedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.c1 = 2.5  # initial cognitive coefficient
        self.c2 = 2.5  # initial social coefficient
        self.min_c1 = 0.5
        self.min_c2 = 0.5
        self.inertia = 0.9  # initial inertia weight
        self.max_inertia = 0.9
        self.min_inertia = 0.4
        self.evaluations = 0
        self.no_improvement_count = 0  # Track number of iterations without improvement
        self.no_improvement_thresh = 5  # Threshold to trigger diversification

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(p) for p in positions])
        self.evaluations += self.swarm_size

        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        while self.evaluations < self.budget:
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            for i in range(self.swarm_size):
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1[i] * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2[i] * (global_best_position - positions[i]))
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
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            diversity = np.mean(np.std(positions, axis=0))

            if self.no_improvement_count >= self.no_improvement_thresh:
                positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
                self.no_improvement_count = 0

            if diversity < (ub - lb).mean() / 10:
                self.inertia = max(self.inertia - 0.05, self.min_inertia)
                self.c1 = max(self.c1 - 0.05, self.min_c1)
                self.c2 = min(self.c2 + 0.1, 4.0)
            else:
                self.inertia = min(self.inertia + 0.05, self.max_inertia)
                self.c1 = min(self.c1 + 0.1, 4.0)
                self.c2 = max(self.c2 - 0.1, self.min_c2)

        return global_best_position, global_best_value