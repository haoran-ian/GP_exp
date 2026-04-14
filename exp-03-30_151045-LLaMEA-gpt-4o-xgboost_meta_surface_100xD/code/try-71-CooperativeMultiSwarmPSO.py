import numpy as np

class CooperativeMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_swarms = 3  # Number of swarms
        self.swarm_size = 50
        self.particles = [np.random.rand(self.swarm_size, dim) for _ in range(self.n_swarms)]
        self.velocities = [np.random.randn(self.swarm_size, dim) for _ in range(self.n_swarms)]
        self.personal_best_positions = [np.copy(p) for p in self.particles]
        self.personal_best_values = [np.full(self.swarm_size, np.inf) for _ in range(self.n_swarms)]
        self.global_best_positions = [np.zeros(dim) for _ in range(self.n_swarms)]
        self.global_best_values = [np.inf for _ in range(self.n_swarms)]
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _inter_swarm_cooperation(self):
        # Share information between swarms
        for i in range(self.n_swarms):
            for j in range(self.n_swarms):
                if i != j:
                    if self.global_best_values[j] < self.global_best_values[i]:
                        self.global_best_positions[i] = self.global_best_positions[j]
                        self.global_best_values[i] = self.global_best_values[j]

    def _adjust_learning_rates(self):
        progress = self.current_eval / self.budget
        self.c1 = self.c1_initial * (1 - progress)
        self.c2 = self.c2_initial * progress

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            self._adjust_learning_rates()
            self._inter_swarm_cooperation()

            for swarm_idx in range(self.n_swarms):
                for i in range(self.swarm_size):
                    value = func(self.particles[swarm_idx][i])
                    self.current_eval += 1

                    if value < self.personal_best_values[swarm_idx][i]:
                        self.personal_best_values[swarm_idx][i] = value
                        self.personal_best_positions[swarm_idx][i] = self.particles[swarm_idx][i]

                    if value < self.global_best_values[swarm_idx]:
                        self.global_best_values[swarm_idx] = value
                        self.global_best_positions[swarm_idx] = self.particles[swarm_idx][i]

                inertia_weight = self._adaptive_inertia_weight()

                for i in range(self.swarm_size):
                    self.velocities[swarm_idx][i] = (
                        inertia_weight * self.velocities[swarm_idx][i] +
                        self.c1 * np.random.rand() * (self.personal_best_positions[swarm_idx][i] - self.particles[swarm_idx][i]) +
                        self.c2 * np.random.rand() * (self.global_best_positions[swarm_idx] - self.particles[swarm_idx][i])
                    )

                    self.particles[swarm_idx][i] += self.velocities[swarm_idx][i]
                    self.particles[swarm_idx][i] = np.clip(self.particles[swarm_idx][i], bounds.lb, bounds.ub)

        best_swarm_idx = np.argmin(self.global_best_values)
        return self.global_best_positions[best_swarm_idx], self.global_best_values[best_swarm_idx]