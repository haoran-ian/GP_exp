import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.c1 = 2.0
        self.c2 = 1.5
        self.inertia_weight = 0.9
        self.mutation_prob = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        phase_switch = self.budget // 3
        for _ in range(self.budget):
            if self.evaluations >= self.budget:
                return self.global_best_position

            if self.evaluations % phase_switch == 0 and self.evaluations > 0:
                # Reduce inertia weight and adjust cognitive and social components as algorithm progresses
                self.inertia_weight = max(0.4, self.inertia_weight - 0.1)
                self.c1 = max(0.5, self.c1 - 0.5)
                self.c2 = min(2.5, self.c2 + 0.5)

            for i in range(self.swarm_size):
                if np.random.rand() < self.mutation_prob:
                    r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = self.positions[r1] + 0.8 * (self.positions[r2] - self.positions[r3])
                    self.positions[i] = np.clip(mutant, lb, ub)

                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                score = func(self.positions[i])
                self.evaluations += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
                
                # Implement local search with varying intensity based on current diversity and success
                if np.random.rand() < 0.1 + 0.2 * (1 - np.mean(self.best_scores < np.inf)):
                    local_pos = self.positions[i] + np.random.uniform(-0.02, 0.02, self.dim)
                    local_pos = np.clip(local_pos, lb, ub)
                    local_score = func(local_pos)
                    self.evaluations += 1
                    if local_score < self.best_scores[i]:
                        self.best_scores[i] = local_score
                        self.best_positions[i] = local_pos.copy()
                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best_position = local_pos.copy()

        return self.global_best_position