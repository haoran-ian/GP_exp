import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.diversity_prob = 0.1
        self.local_search_prob = 0.05

    def levy_flight(self, size, alpha=1.5):
        sigma = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
                 (np.math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v)**(1 / alpha)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        restart_threshold = self.budget // 5
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0 and self.evaluations > 0:
                self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))

            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                success_rate = np.mean(self.best_scores < np.inf)
                self.inertia_weight = 0.4 + 0.5 * success_rate

                progress_ratio = self.evaluations / self.budget
                self.c1 = 2.5 - progress_ratio
                self.c2 = 0.5 + progress_ratio

                phi = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                if np.random.rand() < self.diversity_prob:
                    self.positions[i] += self.levy_flight(self.dim) * (ub - lb)
                    self.positions[i] = np.clip(self.positions[i], lb, ub)

                if np.random.rand() < self.local_search_prob:
                    for d in range(self.dim):
                        local_pos = self.positions[i].copy()
                        local_pos[d] += np.random.uniform(-0.05, 0.05)
                        local_pos = np.clip(local_pos, lb, ub)
                        local_score = func(local_pos)
                        self.evaluations += 1
                        if local_score < self.best_scores[i]:
                            self.best_scores[i] = local_score
                            self.best_positions[i] = local_pos.copy()
                        if local_score < self.global_best_score:
                            self.global_best_score = local_score
                            self.global_best_position = local_pos.copy()

                score = func(self.positions[i])
                self.evaluations += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

        return self.global_best_position