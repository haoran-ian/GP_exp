import numpy as np

class EnhancedPSODE:
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
            self.levy_flight_update(func, lb, ub)

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
                                self.c2 * r2 * (self.global_best_position - self.population[i]))
            self.population[i] += self.velocity[i]
            self.population[i] = np.clip(self.population[i], lb, ub)

    def levy_flight_update(self, func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            step = self.levy_flight(self.dim)
            trial = self.population[i] + step * (self.population[i] - self.global_best_position)
            trial = np.clip(trial, lb, ub)

            trial_fitness = func(trial)
            self.evaluations += 1

            if trial_fitness < self.personal_best_value[i]:
                self.population[i] = trial
                self.personal_best_value[i] = trial_fitness
                self.personal_best_position[i] = trial.copy()

                if trial_fitness < self.global_best_value:
                    self.global_best_value = trial_fitness
                    self.global_best_position = trial.copy()

    def levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        return u / np.abs(v)**(1 / beta)