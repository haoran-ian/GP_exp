import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.initial_c1 = 2.5
        self.initial_c2 = 0.5
        self.final_c1 = 0.5
        self.final_c2 = 2.5
        self.w = 0.5
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_values = np.array([np.inf] * self.pop_size)

    def update_particles(self, lb, ub):
        for i in range(self.pop_size):
            current_c1 = ((self.final_c1 - self.initial_c1) * 
                          (self.evaluations / self.budget) + self.initial_c1)
            current_c2 = ((self.final_c2 - self.initial_c2) * 
                          (self.evaluations / self.budget) + self.initial_c2)
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (self.w * self.velocities[i] +
                                  current_c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                  current_c2 * r2 * (self.global_best - self.population[i]))
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def differential_evolution(self, index, lb, ub):
        idxs = [idx for idx in range(self.pop_size) if idx != index]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.F * (b - c)
        mutant = np.clip(mutant, lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[index])
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                candidate = self.differential_evolution(i, lb, ub)
                candidate_value = func(candidate)
                self.evaluations += 1

                if candidate_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = candidate_value
                    self.personal_best[i] = candidate.copy()

                if candidate_value < self.global_best_value:
                    self.global_best_value = candidate_value
                    self.global_best = candidate.copy()
            
            self.update_particles(lb, ub)

        return self.global_best