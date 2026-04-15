import numpy as np
import scipy.spatial.distance as dist

class AdvancedGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 5
        self.memory = []
        self.evaluations = 0
        self.sharing_threshold = 0.5  # New parameter for fitness sharing
        self.neighborhood_size = 5  # New parameter for adaptive search

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self._initialize_population(bounds)
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            fitness = self._evaluate_fitness(population, func)
            self.evaluations += len(fitness)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx]
                self._update_memory(best_solution)

            selected = self._selection(population, fitness)
            offspring = self._crossover(selected, bounds)
            population = self._mutation(offspring, bounds)

            # Dynamic fitness sharing
            self._fitness_sharing(population, fitness)

            # Adaptive neighborhood-based search
            self._adaptive_neighborhood_search(population, func, bounds)

            # Dynamic parameter adjustment based on progress
            self._adjust_parameters(fitness)

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def _evaluate_fitness(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def _selection(self, population, fitness):
        selected_idx = np.random.choice(np.argsort(fitness)[:self.population_size // 2], size=self.population_size // 2, replace=True)
        return population[selected_idx]

    def _crossover(self, selected, bounds):
        offspring = []
        for i in range(self.population_size // 2):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
                cross_point = np.random.randint(1, self.dim - 1)
                child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                offspring.append(child)
            else:
                offspring.append(selected[i])
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def _mutation(self, offspring, bounds):
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.1 * np.std(offspring, axis=0), self.dim)
                offspring[i] = offspring[i] + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            improvement = np.min([np.linalg.norm(solution - mem) for mem in self.memory])
            if improvement > 0.1:
                self.memory[np.random.randint(0, self.memory_size)] = solution

    def _fitness_sharing(self, population, fitness):
        dist_matrix = dist.squareform(dist.pdist(population))
        shared_fitness = fitness.copy()
        for i in range(len(fitness)):
            niche_count = np.sum(np.exp(-dist_matrix[i] / self.sharing_threshold))
            shared_fitness[i] = fitness[i] * niche_count
        min_shared_idx = np.argmin(shared_fitness)
        self.best_shared_solution = population[min_shared_idx]

    def _adaptive_neighborhood_search(self, population, func, bounds):
        for i in range(self.population_size):
            neighborhood = np.random.normal(loc=population[i], scale=0.1, size=(self.neighborhood_size, self.dim))
            neighborhood = np.clip(neighborhood, bounds[:, 0], bounds[:, 1])
            neighborhood_fitness = np.apply_along_axis(func, 1, neighborhood)
            self.evaluations += len(neighborhood_fitness)
            best_neighborhood_idx = np.argmin(neighborhood_fitness)
            if neighborhood_fitness[best_neighborhood_idx] < func(population[i]):
                population[i] = neighborhood[best_neighborhood_idx]

    def _adjust_parameters(self, fitness):
        improvement = (np.min(fitness) - np.mean(fitness)) / np.std(fitness)
        if improvement < 0.05:
            self.mutation_rate = min(self.mutation_rate * 1.05, 0.5)
            self.crossover_rate = max(self.crossover_rate * 0.95, 0.5)
            self.sharing_threshold = max(self.sharing_threshold * 0.95, 0.1)
        else:
            self.mutation_rate = max(self.mutation_rate * 0.95, 0.05)
            self.crossover_rate = min(self.crossover_rate * 1.05, 0.9)
            self.sharing_threshold = min(self.sharing_threshold * 1.05, 1.0)