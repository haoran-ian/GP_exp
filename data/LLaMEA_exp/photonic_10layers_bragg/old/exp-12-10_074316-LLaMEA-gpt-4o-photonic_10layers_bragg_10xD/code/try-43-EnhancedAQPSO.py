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
        self.adaptive_mutation_base = 0.2
        self.c1 = 2.5
        self.c2 = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        restart_threshold = self.budget // 5
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0 and self.evaluations > 0:
                self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))

            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                progress_ratio = self.evaluations / self.budget
                self.inertia_weight = 0.4 + 0.5 * np.tanh(5 * (1 - progress_ratio))

                self.c1 = 2.5 - progress_ratio
                self.c2 = 0.5 + progress_ratio

                self.velocities[i] *= (0.5 + np.random.random() * 0.5)  # Dynamic velocity scaling

                if np.random.rand() < self.adaptive_mutation_base * (1 - progress_ratio):
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

        return self.global_best_position