import numpy as np

class EnhancedAdaptiveSymbioticEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_rate = 0.5
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.previous_best_fitness = np.inf

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        self.population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size

            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_population = self.population[elite_indices]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.previous_best_fitness = self.best_fitness
                self.best_fitness = fitness[min_idx]
                self.best_solution = self.population[min_idx].copy()

            # Adapt mutation and crossover rates based on convergence speed
            improvement_rate = (self.previous_best_fitness - self.best_fitness) / (np.abs(self.previous_best_fitness) + 1e-8)
            if improvement_rate > 0.01:
                self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
                self.crossover_rate = min(1.0, self.crossover_rate * 1.1)
            else:
                self.mutation_rate = min(1.0, self.mutation_rate * 1.1)
                self.crossover_rate = max(0.1, self.crossover_rate * 0.9)

            parents = self._select_parents(fitness)
            offspring = self._crossover(parents, bounds)
            proximity = np.linalg.norm(self.population - self.best_solution, axis=1)
            mutation_strength_scaler = 1 - (proximity / (np.max(proximity) + 1e-8))
            offspring = self._mutate(offspring, bounds, mutation_strength_scaler)
            
            if elite_count > 0:
                offspring[:elite_count] = elite_population

            # Introduce gradient descent-based local search
            fitness_gradient = np.gradient(fitness)
            offspring[min_idx] -= 0.01 * fitness_gradient[min_idx] 

            self.population = offspring

        return self.best_solution

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