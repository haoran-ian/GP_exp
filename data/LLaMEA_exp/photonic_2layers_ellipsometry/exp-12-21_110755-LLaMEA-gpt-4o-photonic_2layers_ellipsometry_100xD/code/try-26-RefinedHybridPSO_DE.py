import numpy as np

class RefinedHybridPSO_DE:
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
        self.neighborhood_size = 5  # For dynamic topology
        self.diversity_threshold = 1e-5
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf
        self.current_eval = 0

    def adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_values = np.array([np.inf] * self.pop_size)

    def update_particles(self, lb, ub):
        w = self.adaptive_inertia_weight()
        for i in range(self.pop_size):
            neighbors = self.get_neighbors(i)
            local_best = min(neighbors, key=lambda idx: self.personal_best_values[idx])
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                  self.c2 * r2 * (self.personal_best[local_best] - self.population[i]))
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def get_neighbors(self, index):
        indices = list(range(self.pop_size))
        indices.remove(index)
        np.random.shuffle(indices)
        return indices[:self.neighborhood_size]

    def self_adaptive_differential_evolution(self, index, lb, ub):
        F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + F * (b - c)
        mutant = np.clip(mutant, lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[index])
        return trial

    def preserve_diversity(self):
        diversity = np.std(self.population, axis=0)
        if np.all(diversity < self.diversity_threshold):
            self.initialize_population(self.population.min(axis=0), self.population.max(axis=0))
            self.update_best()

    def update_best(self):
        for i in range(self.pop_size):
            value = func(self.population[i])
            if value < self.personal_best_values[i]:
                self.personal_best_values[i] = value
                self.personal_best[i] = self.population[i].copy()
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best = self.population[i].copy()

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.update_best()

        while self.current_eval < self.budget:
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
            self.preserve_diversity()

        return self.global_best