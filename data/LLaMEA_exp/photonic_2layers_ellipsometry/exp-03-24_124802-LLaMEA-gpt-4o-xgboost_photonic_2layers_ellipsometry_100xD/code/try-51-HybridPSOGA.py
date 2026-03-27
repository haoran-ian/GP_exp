import numpy as np

class HybridPSOGA:
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
        self.crossover_prob = 0.7  # Crossover probability for GA
        self.mutation_rate = 1.0 / self.dim  # Mutation rate for GA

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.particles = lb + (ub - lb) * self.particles
        evaluations = 0

        while evaluations < self.budget:
            for i, particle in enumerate(self.particles):
                score = func(particle)
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = particle
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (evaluations / self.budget)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (evaluations / self.budget)
            cognitive = c1 * r1 * (self.personal_best_positions - self.particles)
            social = c2 * r2 * (self.global_best_position - self.particles)
            inertia_weight = 0.4 + np.random.rand() / 2.5
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    parents_indices = np.random.choice(self.population_size, 2, replace=False)
                    parent1, parent2 = self.particles[parents_indices]
                    if np.random.rand() < self.crossover_prob:
                        crossover_point = np.random.randint(1, self.dim)
                        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    else:
                        offspring = np.copy(parent1)

                    mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                    offspring[mutation_mask] = lb + (ub - lb) * np.random.rand(np.sum(mutation_mask))

                    offspring_score = func(offspring)
                    evaluations += 1
                    if offspring_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = offspring_score
                        self.personal_best_positions[i] = offspring

        return self.global_best_position, self.global_best_score