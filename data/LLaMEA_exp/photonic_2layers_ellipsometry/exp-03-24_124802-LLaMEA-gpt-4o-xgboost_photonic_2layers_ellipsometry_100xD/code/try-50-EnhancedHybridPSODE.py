import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F = 0.6
        self.CR = 0.95
        self.c1_max, self.c1_min = 2.5, 0.5
        self.c2_max, self.c2_min = 2.5, 0.5
        self.num_swarms = 5  # Multi-swarm approach
        self.random_walk_prob = 0.1  # Probability for random walk exploration

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.particles = lb + (ub - lb) * self.particles
        evaluations = 0
        swarm_indices = np.array_split(np.arange(self.population_size), self.num_swarms)

        while evaluations < self.budget:
            for swarm_idx in swarm_indices:
                for i in swarm_idx:
                    particle = self.particles[i]
                    score = func(particle)
                    evaluations += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = particle
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = particle

                r1, r2 = np.random.rand(len(swarm_idx), self.dim), np.random.rand(len(swarm_idx), self.dim)
                c1 = self.c1_max - (self.c1_max - self.c1_min) * (evaluations / self.budget)
                c2 = self.c2_min + (self.c2_max - self.c2_min) * (evaluations / self.budget)
                cognitive = c1 * r1 * (self.personal_best_positions[swarm_idx] - self.particles[swarm_idx])
                social = c2 * r2 * (self.global_best_position - self.particles[swarm_idx])
                inertia_weight = 0.4 + np.random.rand() / 2.5
                self.velocities[swarm_idx] = inertia_weight * self.velocities[swarm_idx] + cognitive + social
                self.particles[swarm_idx] += self.velocities[swarm_idx]
                self.particles[swarm_idx] = np.clip(self.particles[swarm_idx], lb, ub)

                for i in swarm_idx:
                    if np.random.rand() < self.random_walk_prob:
                        random_walk = np.random.normal(scale=0.1, size=self.dim)
                        self.particles[i] = np.clip(self.particles[i] + random_walk, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                    x1, x2, x3 = self.particles[indices]
                    F_individual = np.random.uniform(0.4, 0.9)
                    mutant_vector = np.clip(x1 + F_individual * (x2 - x3), lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover, mutant_vector, self.particles[i])
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

        return self.global_best_position, self.global_best_score