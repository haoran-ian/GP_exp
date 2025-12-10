import numpy as np

class EnhancedPSODE_v3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1_min, self.c1_max = 1.0, 2.0
        self.c2_min, self.c2_max = 1.0, 2.0
        self.w_min, self.w_max = 0.4, 0.9
        self.F_min, self.F_max = 0.1, 0.9
        self.CR = 0.9
        self.population = None
        self.velocity = None
        self.personal_best_position = None
        self.personal_best_value = None
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.adaptive_parameters()
            self.update_particles(func, lb, ub)
            self.adaptive_elite_mutation(func, lb, ub)

        return self.global_best_position

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocity = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.array([float('inf')] * self.population_size)

    def adaptive_parameters(self):
        t = self.evaluations / self.budget
        self.c1 = self.c1_min + (self.c1_max - self.c1_min) * (1 - t)
        self.c2 = self.c2_min + (self.c2_max - self.c2_min) * t
        chaos_factor = 0.5 * (1 - np.cos(np.pi * t))**2
        self.w = self.w_min + (self.w_max - self.w_min) * chaos_factor
        self.F = self.F_min + (self.F_max - self.F_min) * (1 - t)
        self.population_size = max(5, int(20 * (1 - t)))

    def update_particles(self, func, lb, ub):
        chaos_perturb = np.random.laplace(0, 0.1, (self.population_size, self.dim))
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            fitness = func(self.population[i])
            self.evaluations += 1

            if fitness < self.personal_best_value[i]:
                self.personal_best_value[i] = fitness
                self.personal_best_position[i] = self.population[i].copy()

            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best_position = self.population[i].copy()

            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocity[i] = (self.w * self.velocity[i] +
                                self.c1 * r1 * (self.personal_best_position[i] - self.population[i]) +
                                self.c2 * r2 * (self.global_best_position - self.population[i]) +
                                chaos_perturb[i] * 0.05)
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def adaptive_elite_mutation(self, func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            elite_index = np.argmin(self.personal_best_value)
            elite = self.personal_best_position[elite_index]

            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            # Dynamic Lévy flight parameter adjustment
            levy_factor = np.random.uniform(0.1, 1.0) * (1 - self.evaluations / self.budget)
            mutant = elite + levy_factor * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, lb, ub)

            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])

            trial_fitness = func(trial)
            self.evaluations += 1

            if trial_fitness < self.personal_best_value[i]:
                self.population[i] = trial
                self.personal_best_value[i] = trial_fitness
                self.personal_best_position[i] = trial.copy()

                if trial_fitness < self.global_best_value:
                    self.global_best_value = trial_fitness
                    self.global_best_position = trial.copy()