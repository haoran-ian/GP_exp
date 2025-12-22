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
        self.inertia_weight = 0.9
        self.phase_switch_threshold = self.budget // 3
        self.c1 = 2.5
        self.c2 = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        for i in range(self.budget):
            if self.evaluations >= self.budget:
                return self.global_best_position

            current_phase = (self.evaluations // self.phase_switch_threshold) % 3
            if current_phase == 0:
                self.inertia_weight *= 0.99
            elif current_phase == 1:
                self.inertia_weight = 0.4 + 0.5 * np.random.rand()
            else:
                self.inertia_weight *= 1.01

            for j in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                phi = np.random.uniform(0, 1, self.dim)
                self.velocities[j] = (self.inertia_weight * self.velocities[j] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[j] - self.positions[j]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[j]))
                self.positions[j] += self.velocities[j]

                self.positions[j] = np.clip(self.positions[j], lb, ub)

                # Crowding-based diversity mechanism
                distance_to_gbest = np.linalg.norm(self.positions[j] - self.global_best_position)
                if np.random.rand() < min(0.2, max(0, 1 - distance_to_gbest / (ub - lb).mean())):
                    self.positions[j] = np.random.uniform(lb, ub, self.dim)
                
                # Evaluate the fitness
                score = func(self.positions[j])
                self.evaluations += 1

                if score < self.best_scores[j]:
                    self.best_scores[j] = score
                    self.best_positions[j] = self.positions[j].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[j].copy()
                
        return self.global_best_position