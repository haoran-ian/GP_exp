import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.c1 = 1.2
        self.c2 = 1.8
        self.F = 0.7
        self.CR_max = 0.9
        self.CR_min = 0.75
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

    def update_personal_best(self, fitness, func):
        for i in range(self.population_size):
            personal_fitness = func(self.best_positions[i])
            if fitness[i] < personal_fitness:
                self.best_positions[i] = self.population[i].copy()

    def update_global_best(self, fitness):
        min_index = np.argmin(fitness)
        if fitness[min_index] < self.global_best_fitness:
            self.global_best_fitness = fitness[min_index]
            self.global_best_position = self.population[min_index].copy()

    def adaptive_parameters(self):
        inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * (self.eval_count / self.budget))
        CR = self.CR_max - ((self.CR_max - self.CR_min) * (self.eval_count / self.budget))
        return inertia_weight, CR

    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.best_positions - self.population)
        social_component = self.c2 * r2 * (self.global_best_position - self.population)
        
        inertia_weight, _ = self.adaptive_parameters()
        self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
        self.population += self.velocities
        self.population = np.clip(self.population, lb, ub)

    def differential_evolution(self, bounds, fitness, func):
        lb, ub = bounds.lb, bounds.ub
        new_population = self.population.copy()
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, lb, ub)
            crossover = np.random.rand(self.dim) < self.adaptive_parameters()[1]
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
        self.update_personal_best(fitness, func)
        self.update_global_best(fitness)

        while self.eval_count < self.budget:
            self.update_velocities_and_positions(func_bounds)
            self.differential_evolution(func_bounds, fitness, func)
            fitness = self.evaluate_population(func)
            self.update_personal_best(fitness, func)
            self.update_global_best(fitness)

        return self.global_best_position