import numpy as np

class RefinedAQPSO:
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

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        restart_threshold = self.budget // 5
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0 and self.evaluations > 0:
                self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
            
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                # Adaptive inertia weight based on convergence
                success_rate = np.mean(self.best_scores < np.inf)
                # Change 1: Use a sigmoid function for inertia weight adaptation
                self.inertia_weight = 0.4 + 0.5 / (1 + np.exp(-10 * (success_rate - 0.5)))

                # Dynamic update for c1 and c2
                progress_ratio = self.evaluations / self.budget
                self.c1 = 2.5 - progress_ratio
                self.c2 = 0.5 + progress_ratio

                phi = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]

                self.positions[i] = np.clip(self.positions[i], lb, ub)

                # Introduce diversity-preservation mechanism
                if np.random.rand() < self.diversity_prob:
                    self.positions[i] = np.random.uniform(lb, ub, self.dim)
                
                # Local search intensification
                if np.random.rand() < self.local_search_prob:
                    # Change 2: Adjust position perturbation for local search
                    local_pos = self.positions[i] + np.random.uniform(-0.1, 0.1, self.dim)
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