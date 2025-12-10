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

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        restart_threshold = self.budget // 5  # 1 change: added restart mechanism
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0:
                # 2 change: stochastic restarts to improve diversity
                self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))            
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                # Adaptive inertia weight based on improvement rate
                if self.evaluations > 0:
                    improvement_rate = (self.global_best_score - min(self.best_scores)) / self.global_best_score
                    self.inertia_weight = 0.5 + 0.4 * improvement_rate

                phi = np.random.uniform(0, 1, self.dim)
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