import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR = 0.9
        self.restart_threshold = 0.1 * budget
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf
        self.current_eval = 0
        self.pop_size = self.initial_pop_size

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
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[index])
        return trial

    def restart_population(self, lb, ub):
        if self.current_eval < self.budget - self.restart_threshold:
            self.pop_size = np.random.randint(self.initial_pop_size // 2, self.initial_pop_size * 2)
            self.initialize_population(lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
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
            # Restart mechanism
            if self.current_eval % (self.budget // 10) == 0:
                self.restart_population(lb, ub)

        return self.global_best