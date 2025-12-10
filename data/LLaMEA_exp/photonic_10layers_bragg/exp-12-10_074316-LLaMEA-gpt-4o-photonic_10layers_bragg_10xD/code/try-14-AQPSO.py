import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.weights = np.random.rand(self.swarm_size, self.dim)
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.neighborhood_size = 5  # Number of neighbors to consider

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        restart_threshold = self.budget // 5
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0:
                self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                # Dynamic inertia weight based on personal best improvement
                if self.best_scores[i] < np.inf:
                    self.inertia_weight = 0.5 + 0.4 * (1 - (self.best_scores[i] / (self.global_best_score + 1e-10)))

                # Dynamic update for c1 and c2
                progress_ratio = self.evaluations / self.budget
                self.c1 = 2.5 - progress_ratio  # Decreases over time
                self.c2 = 0.5 + progress_ratio  # Increases over time

                # Neighborhood-based learning
                neighbors = np.random.choice(self.swarm_size, self.neighborhood_size, replace=False)
                neighborhood_best_position = min(neighbors, key=lambda n: self.best_scores[n])
                phi = np.random.uniform(0, 1, self.dim)
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.best_positions[neighborhood_best_position] - self.positions[i]))
                
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