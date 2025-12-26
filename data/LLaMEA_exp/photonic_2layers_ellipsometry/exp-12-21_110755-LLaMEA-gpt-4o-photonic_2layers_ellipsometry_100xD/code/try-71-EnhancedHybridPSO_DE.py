import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR = 0.9
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf
        self.current_eval = 0
        self.elite_preservation_ratio = 0.1

    def adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def adaptive_learning_factors(self):
        progress = self.current_eval / self.budget
        self.c1 = 2.5 - 1.5 * progress  # Decrease c1 over time
        self.c2 = 0.5 + 1.5 * progress  # Increase c2 over time

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_values = np.array([np.inf] * self.pop_size)

    def update_particles(self, lb, ub):
        w = self.adaptive_inertia_weight()
        self.adaptive_learning_factors()
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                  self.c2 * r2 * (self.global_best - self.population[i]))
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def self_adaptive_differential_evolution(self, index, lb, ub):
        F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
        mutant = np.clip(mutant, lb, ub)
        self.CR = 0.5 + 0.4 * np.cos(np.pi * self.current_eval / self.budget) 
        self.CR = np.clip(self.CR, 0.5, 1.0)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[index])
        return trial

    def elite_preservation(self):
        elite_size = int(self.pop_size * self.elite_preservation_ratio)
        elite_indices = np.argsort(self.personal_best_values)[:elite_size]
        return elite_indices

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        while self.current_eval < self.budget:
            elite_indices = self.elite_preservation()
            for i in range(self.pop_size):
                candidate = self.self_adaptive_differential_evolution(i, lb, ub)
                candidate_value = func(candidate)
                self.current_eval += 1

                if candidate_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = candidate_value
                    self.personal_best[i] = candidate.copy()

                if candidate_value < self.global_best_value:
                    self.global_best_value = candidate_value
                    self.global_best = candidate.copy()

            self.update_particles(lb, ub)
            for i in elite_indices:
                self.personal_best[i] = self.population[i].copy()

        return self.global_best