import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 30
        self.swarm_size = self.initial_swarm_size
        self.positions = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.diversity_prob = 0.1
        self.local_search_prob = 0.05
        self.c1 = 2.5
        self.c2 = 0.5
        self.mutation_prob = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        adaptive_bounds = np.array([lb, ub]).T
        restart_threshold = self.budget // 5
        
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0 and self.evaluations > 0:
                self.positions = np.random.uniform(adaptive_bounds[:, 0], adaptive_bounds[:, 1], (self.swarm_size, self.dim))
                # Dynamically adjust swarm size based on performance
                performance_metric = (self.global_best_score - np.mean(self.best_scores)) / self.global_best_score
                self.swarm_size = max(10, int(self.initial_swarm_size * (1 + performance_metric)))
                self.positions = np.resize(self.positions, (self.swarm_size, self.dim))
                self.velocities = np.resize(self.velocities, (self.swarm_size, self.dim))
                self.best_positions = np.resize(self.best_positions, (self.swarm_size, self.dim))
                self.best_scores = np.resize(self.best_scores, self.swarm_size)
                self.best_scores.fill(np.inf)
                
            success_rate = np.mean(self.best_scores < np.inf)
            self.diversity_prob = 0.1 + 0.2 * (1 - success_rate)
            self.local_search_prob = 0.05 + 0.1 * success_rate
            self.mutation_prob = 0.1 + 0.2 * success_rate

            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                progress_ratio = self.evaluations / self.budget
                self.inertia_weight = 0.4 + 0.5 * np.tanh(5 * (1 - progress_ratio))

                self.c1 = 2.5 - progress_ratio
                self.c2 = 0.5 + progress_ratio

                if np.random.rand() < self.mutation_prob:
                    r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                    mutant = self.positions[r1] + 0.8 * (self.positions[r2] - self.positions[r3])
                    self.positions[i] = np.clip(mutant, adaptive_bounds[:, 0], adaptive_bounds[:, 1])

                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], adaptive_bounds[:, 0], adaptive_bounds[:, 1])

                if np.random.rand() < self.diversity_prob:
                    self.positions[i] = np.random.uniform(adaptive_bounds[:, 0], adaptive_bounds[:, 1], self.dim)
                
                if np.random.rand() < self.local_search_prob:
                    local_pos = self.positions[i] + np.random.uniform(-0.02, 0.02, self.dim)
                    local_pos = np.clip(local_pos, adaptive_bounds[:, 0], adaptive_bounds[:, 1])
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

            # Update adaptive bounds based on current best and worst positions
            if self.evaluations % restart_threshold == 0:
                best_pos = self.positions[np.argmin(self.best_scores)]
                worst_pos = self.positions[np.argmax(self.best_scores)]
                adaptive_bounds[:, 0] = np.minimum(lb, best_pos - 0.1 * (best_pos - worst_pos))
                adaptive_bounds[:, 1] = np.maximum(ub, best_pos + 0.1 * (worst_pos - best_pos))
                
        return self.global_best_position