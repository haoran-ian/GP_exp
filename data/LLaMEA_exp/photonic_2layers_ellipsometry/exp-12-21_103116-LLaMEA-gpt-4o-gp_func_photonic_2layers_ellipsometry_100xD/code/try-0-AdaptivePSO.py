import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.inertia = 0.5  # inertia weight
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize swarm particles and velocities
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.array([func(p) for p in positions])
        self.evaluations += self.swarm_size

        # Initialize global best
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        while self.evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.swarm_size):
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # Evaluate current position
                current_value = func(positions[i])
                self.evaluations += 1

                # Update personal bests
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_values[i] = current_value

            # Update global best
            current_best_index = np.argmin(personal_best_values)
            current_best_value = personal_best_values[current_best_index]
            if current_best_value < global_best_value:
                global_best_position = personal_best_positions[current_best_index]
                global_best_value = current_best_value

            # Adaptive inertia weight - increase if swarm is converged, decrease if diverse
            diversity = np.mean(np.std(positions, axis=0))
            if diversity < (ub - lb).mean() / 10:
                self.inertia = min(self.inertia + 0.05, 1.0)
            else:
                self.inertia = max(self.inertia - 0.05, 0.1)

        return global_best_position, global_best_value