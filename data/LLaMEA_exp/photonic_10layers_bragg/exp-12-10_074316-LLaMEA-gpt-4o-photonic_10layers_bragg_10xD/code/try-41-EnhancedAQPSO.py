import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 30
        self.positions = np.random.uniform(0, 1, (self.initial_swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.initial_swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.initial_swarm_size, np.inf)
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
        restart_threshold = self.budget // 5
        swarm_size = self.initial_swarm_size
        evaluation_interval = self.budget // 10
        
        for _ in range(self.budget):
            if self.evaluations % restart_threshold == 0 and self.evaluations > 0:
                self.positions = np.random.uniform(lb, ub, (swarm_size, self.dim))
            
            progress_ratio = self.evaluations / self.budget
            self.inertia_weight = 0.4 + 0.5 * np.tanh(5 * (1 - progress_ratio))
            self.c1 = 2.5 - progress_ratio
            self.c2 = 0.5 + progress_ratio

            if self.evaluations > 0 and self.evaluations % evaluation_interval == 0:
                improvement_rate = (self.global_best_score - np.min(self.best_scores)) / self.global_best_score
                if improvement_rate < 0.01:
                    swarm_size = min(swarm_size + 10, 100)
                    self.positions = np.vstack((self.positions, np.random.uniform(lb, ub, (10, self.dim))))
                    self.velocities = np.vstack((self.velocities, np.random.uniform(-0.1, 0.1, (10, self.dim))))
                    self.best_positions = np.vstack((self.best_positions, np.random.uniform(lb, ub, (10, self.dim))))
                    self.best_scores = np.append(self.best_scores, np.full(10, np.inf))
                self.mutation_prob = 0.1 + 0.2 * (1 - improvement_rate)
            
            for i in range(swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                if np.random.rand() < self.mutation_prob:
                    r1, r2, r3 = np.random.choice(swarm_size, 3, replace=False)
                    mutant = self.positions[r1] + 0.8 * (self.positions[r2] - self.positions[r3])
                    self.positions[i] = np.clip(mutant, lb, ub)

                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                if np.random.rand() < self.diversity_prob:
                    self.positions[i] = np.random.uniform(lb, ub, self.dim)
                
                if np.random.rand() < self.local_search_prob:
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

                score = func(self.positions[i])
                self.evaluations += 1

                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
                
        return self.global_best_position