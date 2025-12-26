import numpy as np

class ImprovedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_decay = 0.99
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.c1_final = 0.5
        self.c2_final = 2.5
        self.F_initial = 0.8
        self.F_decay = 0.99
        self.CR = 0.9
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.eval_count = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocities = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.population_size, self.dim))
        self.best_positions = self.population.copy()
        self.global_best_position = self.population[0].copy()

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        self.eval_count += self.population_size
        return fitness

    def update_personal_best(self, fitness):
        for i in range(self.population_size):
            if fitness[i] < func(self.best_positions[i]):
                self.best_positions[i] = self.population[i].copy()

    def update_global_best(self, fitness):
        min_index = np.argmin(fitness)
        if fitness[min_index] < self.global_best_fitness:
            self.global_best_fitness = fitness[min_index]
            self.global_best_position = self.population[min_index].copy()

    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        c1 = self.c1_initial - (self.c1_initial - self.c1_final) * (self.eval_count / self.budget)
        c2 = self.c2_initial + (self.c2_final - self.c2_initial) * (self.eval_count / self.budget)
        cognitive_component = c1 * r1 * (self.best_positions - self.population)
        social_component = c2 * r2 * (self.global_best_position - self.population)

        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
        self.inertia_weight *= self.inertia_decay
        self.population += self.velocities
        self.population = np.clip(self.population, lb, ub)

    def differential_evolution(self, bounds, fitness, func):
        lb, ub = bounds.lb, bounds.ub
        new_population = self.population.copy()
        F = self.F_initial * self.F_decay ** (self.eval_count / self.budget)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + F * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, lb, ub)
            crossover = np.random.rand(self.dim) < self.CR
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.eval_count += 1
            if trial_fitness < fitness[i]:
                new_population[i] = trial
        self.population = new_population

    def __call__(self, func):
        func_bounds = func.bounds
        self.initialize_population(func_bounds)
        fitness = self.evaluate_population(func)
        self.update_personal_best(fitness)
        self.update_global_best(fitness)

        while self.eval_count < self.budget:
            self.update_velocities_and_positions(func_bounds)
            self.differential_evolution(func_bounds, fitness, func)
            fitness = self.evaluate_population(func)
            self.update_personal_best(fitness)
            self.update_global_best(fitness)

        return self.global_best_position