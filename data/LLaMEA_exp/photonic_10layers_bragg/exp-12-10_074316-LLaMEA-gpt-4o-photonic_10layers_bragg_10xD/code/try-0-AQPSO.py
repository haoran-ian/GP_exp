import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.weights = np.random.rand(self.swarm_size, self.dim)
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                # Calculate quantum-inspired movement
                phi = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = phi * self.velocities[i] + self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) + \
                                     self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
                self.positions[i] += self.velocities[i]
                
                # Handle position bounds
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                # Evaluate the function
                score = func(self.positions[i])
                self.evaluations += 1

                # Update personal and global bests
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
                
        return self.global_best_position