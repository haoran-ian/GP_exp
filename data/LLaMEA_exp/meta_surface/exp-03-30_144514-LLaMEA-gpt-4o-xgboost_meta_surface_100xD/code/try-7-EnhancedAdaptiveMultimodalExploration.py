import numpy as np

class EnhancedAdaptiveMultimodalExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(50, self.budget // 5)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptation_threshold = 0.05
        self.elitism_rate = 0.1

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_best(self, population, fitness):
        idx = np.argsort(fitness)
        elite_size = max(int(self.population_size * self.elitism_rate), 1)
        return population[idx][:elite_size]

    def dynamic_mutation(self, fitness):
        min_fit, max_fit = np.min(fitness), np.max(fitness)
        if max_fit > min_fit:
            self.mutation_factor = 0.4 + 0.6 * (max_fit - fitness) / (max_fit - min_fit)
        return self.mutation_factor

    def differential_evolution(self, population, bounds):
        offspring = []
        for i in range(len(population)):
            x_t = population[i]
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor[i] * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            child = np.where(cross_points, mutant, x_t)
            offspring.append(child)
        return np.array(offspring)

    def stochastic_tunneling(self, fitness):
        min_fit = np.min(fitness)
        adjusted_fitness = np.exp(-fitness + min_fit)
        return adjusted_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            adjusted_fitness = self.stochastic_tunneling(fitness)
            self.dynamic_mutation(adjusted_fitness)
            parents = self.select_best(population, adjusted_fitness)
            self.mutation_factor = self.dynamic_mutation(adjusted_fitness)
            population = self.differential_evolution(parents, bounds)
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]