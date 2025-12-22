import numpy as np

class QuantumEnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.w_min, self.w_max = 0.4, 0.9
        self.population = None
        self.velocity = None
        self.personal_best_position = None
        self.personal_best_value = None
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.evaluations = 0
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.adaptive_parameters()
            self.update_particles(func, lb, ub)
            self.chaotic_mutation(func, lb, ub)

        return self.global_best_position

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocity = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.array([float('inf')] * self.population_size)

    def adaptive_parameters(self):
        t = self.evaluations / self.budget
        self.c1 = self.c1_min + (self.c1_max - self.c1_min) * (1 - t**2)
        self.c2 = self.c2_min + (self.c2_max - self.c2_min) * t**2
        self.w = self.w_max * (1 - t) + self.w_min * t
        self.population_size = max(5, int(50 * (1 - t / 2)))

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

    def chaotic_mutation(self, func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            elite_index = np.argmin(self.personal_best_value)
            elite = self.personal_best_position[elite_index]

            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)

            # Chaotic Differential Mutation
            beta = 0.5 + 0.3 * np.cos(3 * np.pi * self.evaluations / self.budget)
            mutant = elite + beta * (self.population[a] - self.population[b])
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