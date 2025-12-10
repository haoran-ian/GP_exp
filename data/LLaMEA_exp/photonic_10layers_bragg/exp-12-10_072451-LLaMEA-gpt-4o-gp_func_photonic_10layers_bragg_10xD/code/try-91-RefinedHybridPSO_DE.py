import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.inertia_weight = 0.7
        self.c1 = 1.2
        self.c2 = 1.8
        self.F = 0.7
        self.CR = 0.85
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.eval_count = 0
        self.rank_probability = 0.45  # Probability for stochastic ranking

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

    def update_velocities_and_positions(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.best_positions - self.population)
        social_component = self.c2 * r2 * (self.global_best_position - self.population)

        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
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
            crossover = np.random.rand(self.dim) < self.CR
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.eval_count += 1
            if trial_fitness < fitness[i]:
                new_population[i] = trial
        self.population = new_population

    def stochastic_ranking(self, fitness):
        ranks = np.argsort(fitness)
        for i in range(self.population_size - 1):
            for j in range(self.population_size - 1, i, -1):
                if np.random.rand() < self.rank_probability:
                    if fitness[ranks[j]] < fitness[ranks[j - 1]]:
                        ranks[j], ranks[j - 1] = ranks[j - 1], ranks[j]
                else:
                    if fitness[ranks[j]] < fitness[ranks[j - 1]]:
                        ranks[j], ranks[j - 1] = ranks[j - 1], ranks[j]
        return ranks

    def dynamic_parameter_adjustment(self):
        self.inertia_weight *= 0.99
        self.c1 += (np.random.rand() - 0.5) * 0.1
        self.c2 += (np.random.rand() - 0.5) * 0.1
        self.F += (np.random.rand() - 0.5) * 0.1
        self.CR += (np.random.rand() - 0.5) * 0.1

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
            self.stochastic_ranking(fitness)
            self.dynamic_parameter_adjustment()

        return self.global_best_position