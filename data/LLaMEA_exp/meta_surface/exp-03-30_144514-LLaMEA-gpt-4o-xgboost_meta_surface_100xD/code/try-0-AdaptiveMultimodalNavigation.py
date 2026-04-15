import numpy as np

class AdaptiveMultimodalNavigation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(50, self.budget // 5)  # Base population size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.adaptation_threshold = 0.05

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness

    def select_parents(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx][:self.population_size // 2]

    def crossover(self, parents):
        offspring = []
        for _ in range(len(parents)):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                cross_point = np.random.randint(1, self.dim - 1)
                child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                offspring.append(child)
            else:
                offspring.append(parents[np.random.randint(len(parents))])
        return np.array(offspring)

    def mutate(self, offspring, bounds):
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.randn(self.dim) * (bounds.ub - bounds.lb) * 0.1
                offspring[i] = np.clip(offspring[i] + mutation_vector, bounds.lb, bounds.ub)
        return offspring

    def adapt_population_size(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < self.adaptation_threshold:
            self.population_size = min(self.population_size * 2, self.budget)
        else:
            self.population_size = max(self.population_size // 2, 10)

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        while self.evaluations < self.budget:
            fitness = self.evaluate_population(population, func)
            self.adapt_population_size(fitness)
            parents = self.select_parents(population, fitness)
            offspring = self.crossover(parents)
            population = self.mutate(offspring, bounds)
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]