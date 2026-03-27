import numpy as np

class AdvancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.subpopulation_size = max(5, self.population_size // 5)
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.temporal_learning_rate = 0.1
        self.F = 0.6
        self.CR = 0.95
        self.c1_max, self.c1_min = 2.5, 0.5
        self.c2_max, self.c2_min = 2.5, 0.5

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
            temporal_learning = self.temporal_learning_rate * (self.global_best_position - self.particles)
            inertia_weight = 0.4 + np.random.rand() / 2.5
            self.velocities = inertia_weight * self.velocities + cognitive + social + temporal_learning
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(0, self.population_size, self.subpopulation_size):
                    subpopulation_indices = np.arange(i, min(i + self.subpopulation_size, self.population_size))
                    for j in subpopulation_indices:
                        indices = np.random.choice([k for k in subpopulation_indices if k != j], 3, replace=False)
                        x1, x2, x3 = self.particles[indices]
                        F_individual = np.random.uniform(0.4, 0.9)
                        mutant_vector = np.clip(x1 + F_individual * (x2 - x3), lb, ub)
                        crossover = np.random.rand(self.dim) < self.CR
                        trial_vector = np.where(crossover, mutant_vector, self.particles[j])
                        trial_score = func(trial_vector)
                        evaluations += 1
                        if trial_score < self.personal_best_scores[j]:
                            self.personal_best_scores[j] = trial_score
                            self.personal_best_positions[j] = trial_vector

        return self.global_best_position, self.global_best_score