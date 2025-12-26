import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 30
        self.current_swarm_size = self.initial_swarm_size
        self.positions = np.random.uniform(0, 1, (self.current_swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.current_swarm_size, self.dim))
        self.best_positions = self.positions.copy()
        self.best_scores = np.full(self.current_swarm_size, np.inf)
        self.global_best_position = np.random.uniform(0, 1, self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0
        self.diversity_prob = 0.1
        self.local_search_prob = 0.05
        self.c1 = 2.5
        self.c2 = 0.5
        self.mutation_prob = 0.1
        self.chaotic_factor = 0.7

    def logistic_map(self, x):
        return 4 * x * (1 - x)
    
    def levy_flight(self, lam=1.5):
        u = np.random.normal(0, 1, self.dim) * (np.sin(np.pi * lam / 2) / np.pi)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / lam))
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        for _ in range(self.budget):
            if self.evaluations % (self.budget // 10) == 0 and self.evaluations > 0:
                self.current_swarm_size = np.random.randint(20, 40)
                self.positions = np.random.uniform(lb, ub, (self.current_swarm_size, self.dim))
                self.velocities = np.random.uniform(-0.1, 0.1, (self.current_swarm_size, self.dim))
                self.best_positions = self.positions.copy()
                self.best_scores = np.full(self.current_swarm_size, np.inf)
            
            success_rate = np.mean(self.best_scores < np.inf)
            self.diversity_prob = 0.1 + 0.2 * (1 - success_rate)
            self.local_search_prob = 0.05 + 0.1 * (1 - self.global_best_score / np.max(self.best_scores))
            self.mutation_prob = 0.1 + 0.2 * success_rate

            # Modify the chaotic factor based on the success rate to enhance exploration
            self.chaotic_factor = self.logistic_map(self.chaotic_factor * (1 + 0.1 * success_rate))

            for i in range(self.current_swarm_size):
                if self.evaluations >= self.budget:
                    return self.global_best_position

                self.inertia_weight = 0.4 + 0.5 * self.chaotic_factor
                progress_ratio = self.evaluations / self.budget

                self.c1 = 2.5 - progress_ratio * self.chaotic_factor
                self.c2 = 0.5 + progress_ratio * self.chaotic_factor

                if np.random.rand() < self.mutation_prob:
                    r1, r2, r3 = np.random.choice(self.current_swarm_size, 3, replace=False)
                    scale = 0.8 * self.chaotic_factor
                    mutant = self.positions[r1] + scale * (self.positions[r2] - self.positions[r3])
                    self.positions[i] = np.clip(mutant, lb, ub)

                damping_factor = 0.9 + 0.1 * self.chaotic_factor
                self.velocities[i] = (damping_factor * self.inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i]) +
                                      self.c2 * np.random.rand(self.dim) * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                if np.random.rand() < self.diversity_prob:
                    self.positions[i] = np.random.uniform(lb, ub, self.dim)

                if np.random.rand() < self.local_search_prob:
                    step = self.levy_flight()
                    local_pos = self.positions[i] + step
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