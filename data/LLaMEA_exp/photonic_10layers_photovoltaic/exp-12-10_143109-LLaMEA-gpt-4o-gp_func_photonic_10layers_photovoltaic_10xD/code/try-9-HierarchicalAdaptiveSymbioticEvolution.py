import numpy as np
from sklearn.cluster import KMeans

class HierarchicalAdaptiveSymbioticEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_rate = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        self.population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_fitness = fitness[min_idx]
                self.best_solution = self.population[min_idx].copy()

            if evaluations > self.budget // 10:
                avg_fitness = np.mean(fitness)
                fitness_std = np.std(fitness)
                self._adapt_mutation_rate(avg_fitness, fitness_std)
                self._adapt_crossover_rate()
                self._adjust_population_size(fitness_std, evaluations)
                self._strategy_adaptation(fitness)

            parents = self._select_parents(fitness)
            offspring = self._crossover(parents, bounds)
            mutation_strength_scaler = self._calculate_mutation_strength_scaler()
            offspring = self._mutate(offspring, bounds, mutation_strength_scaler)

            self.population = offspring

        return self.best_solution

    def _adapt_mutation_rate(self, avg_fitness, fitness_std):
        if self.best_fitness < avg_fitness - 0.1 * fitness_std:
            self.mutation_rate = min(1.0, self.mutation_rate + 0.1)
        else:
            self.mutation_rate = max(0.1, self.mutation_rate - 0.1)

    def _adapt_crossover_rate(self):
        diversity = np.std(self.population, axis=0).mean()
        self.crossover_rate = 0.5 + 0.5 * (diversity / np.max(np.std(self.population, axis=0)))

    def _adjust_population_size(self, fitness_std, evaluations):
        if fitness_std < 0.1 and self.population_size > 5:
            self.population_size = max(5, self.population_size - 1)
        elif fitness_std > 0.5 and evaluations + self.population_size <= self.budget:
            self.population_size = min(100, self.population_size + 1)

    def _strategy_adaptation(self, fitness):
        fitness_gradient = np.gradient(fitness)
        if np.mean(fitness_gradient) < 0:
            self.mutation_rate *= 1.1
            self.crossover_rate *= 0.9

    def _calculate_mutation_strength_scaler(self):
        proximity = np.linalg.norm(self.population - self.best_solution, axis=1)
        mutation_strength_scaler = 1 - (proximity / np.max(proximity))
        return mutation_strength_scaler

    def _select_parents(self, fitness):
        probabilities = 1.0 / (1.0 + fitness)
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.population), self.population_size, p=probabilities)
        return self.population[indices]

    def _crossover(self, parents, bounds):
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                offspring[i][:crossover_point] = parents[i][:crossover_point]
                offspring[i][crossover_point:] = parents[i+1][crossover_point:]
                offspring[i+1][:crossover_point] = parents[i+1][:crossover_point]
                offspring[i+1][crossover_point:] = parents[i][crossover_point:]
            else:
                offspring[i], offspring[i+1] = parents[i], parents[i+1]
        return np.clip(offspring, bounds[0], bounds[1])

    def _mutate(self, offspring, bounds, mutation_strength_scaler):
        mutation_strength = (bounds[1] - bounds[0]) * self.mutation_rate
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate * mutation_strength_scaler[i]:
                offspring[i] += np.random.normal(0, mutation_strength * mutation_strength_scaler[i], self.dim)
        return np.clip(offspring, bounds[0], bounds[1])