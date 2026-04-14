import numpy as np
from scipy.spatial import distance

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 5
        self.memory = []
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self._initialize_population(bounds)
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, population)
            self.evaluations += len(fitness)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx]
                self._update_memory(best_solution)

            clusters = self._cluster_solutions(population, fitness)
            selected = self._selection(population, fitness, clusters)
            offspring = self._multi_parent_crossover(selected, bounds)
            population = self._adaptive_mutation(offspring, bounds)

            # Dynamic parameter adjustment based on progress
            self._adjust_parameters(fitness)

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def _selection(self, population, fitness, clusters):
        selected = []
        for cluster in clusters:
            if len(cluster) > 0:
                cluster_indices = np.array(cluster)
                cluster_fitness = fitness[cluster_indices]
                best_in_cluster_idx = np.argmin(cluster_fitness)
                selected.append(population[cluster_indices[best_in_cluster_idx]])
        return np.array(selected)

    def _multi_parent_crossover(self, selected, bounds):
        offspring = []
        num_parents = min(len(selected), 3)
        for _ in range(self.population_size):
            if np.random.rand() < self.crossover_rate:
                parents = selected[np.random.choice(len(selected), num_parents, replace=False)]
                child = np.mean(parents, axis=0)
                offspring.append(child)
            else:
                offspring.append(selected[np.random.choice(len(selected))])
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def _adaptive_mutation(self, offspring, bounds):
        diversity = np.mean(distance.pdist(offspring))
        adaptive_mutation_rate = max(0.05, min(0.5, diversity))
        for i in range(len(offspring)):
            if np.random.rand() < adaptive_mutation_rate:
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                offspring[i] = offspring[i] + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            self.memory[np.random.randint(0, self.memory_size)] = solution

    def _adjust_parameters(self, fitness):
        improvement = (np.min(fitness) - np.mean(fitness)) / np.std(fitness)
        if improvement < 0.05:
            self.mutation_rate = min(self.mutation_rate * 1.05, 0.5)
            self.crossover_rate = max(self.crossover_rate * 0.95, 0.5)
        else:
            self.mutation_rate = max(self.mutation_rate * 0.95, 0.05)
            self.crossover_rate = min(self.crossover_rate * 1.05, 0.9)

    def _cluster_solutions(self, population, fitness):
        # Simple clustering based on distance
        clusters = []
        threshold = 0.1 * np.mean(np.std(population, axis=0))
        for i, solution in enumerate(population):
            added_to_cluster = False
            for cluster in clusters:
                if np.all(distance.euclidean(solution, population[cluster[0]]) < threshold):
                    cluster.append(i)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([i])
        return clusters