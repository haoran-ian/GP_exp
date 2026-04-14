import numpy as np

class DynamicAdaptiveMultimodalExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.initial_population_size = min(50, self.budget // 5)
        self.population_size = self.initial_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.dynamic_scale = 1.0
        self.stagnation_counter = 0
        self.stagnation_limit = 5

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx][:self.population_size // 2]

    def differential_evolution(self, population, bounds):
        offspring = []
        for i in range(len(population)):
            x_t = population[i]
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * self.dynamic_scale * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < (self.crossover_rate * self.dynamic_scale)
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            child = np.where(cross_points, mutant, x_t)
            offspring.append(child)
        return np.array(offspring)

    def stochastic_tunneling(self, fitness):
        min_fit = np.min(fitness)
        adjusted_fitness = np.exp(-fitness + min_fit)
        return adjusted_fitness

    def adapt_population_diversity(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < 0.05:
            self.dynamic_scale = min(self.dynamic_scale * 1.1, 2.0)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.stagnation_limit:
                self.dynamic_scale = max(self.dynamic_scale * 0.9, 0.5)
                self.stagnation_counter = 0

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            adjusted_fitness = self.stochastic_tunneling(fitness)
            self.adapt_population_diversity(adjusted_fitness)
            parents = self.select_best(population, adjusted_fitness)
            population = self.differential_evolution(parents, bounds)
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]