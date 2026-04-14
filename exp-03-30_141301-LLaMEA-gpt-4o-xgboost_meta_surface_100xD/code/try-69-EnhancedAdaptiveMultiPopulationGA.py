import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

class EnhancedAdaptiveMultiPopulationGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.subpopulations = 4
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 15
        self.memory = []
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        populations = [self._initialize_population(bounds) for _ in range(self.subpopulations)]
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for idx, population in enumerate(populations):
                fitness = np.apply_along_axis(func, 1, population)
                self.evaluations += len(fitness)
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_fitness = fitness[best_idx]
                    best_solution = population[best_idx]
                    self._update_memory(best_solution)

                reduced_population = self._dimensionality_reduction(population)
                selected = self._selection(reduced_population, fitness)
                offspring = self._crossover(selected, bounds)
                population = self._mutation(offspring, bounds, idx)

                best_solution = self._adaptive_local_search(best_solution, func, bounds)
                self._adaptive_memory_management(population, fitness)
                self._adaptive_parameter_adjustment(population, fitness)

                populations[idx] = population

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

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

    def _mutation(self, offspring, bounds, subpop_idx):
        dist_matrix = dist.squareform(dist.pdist(offspring))
        clustering = AgglomerativeClustering(n_clusters=min(5, len(offspring) // 2), affinity='precomputed', linkage='average')
        clusters = clustering.fit_predict(dist_matrix)

        for i, individual in enumerate(offspring):
            if np.random.rand() < self.mutation_rate:
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 1:
                    mutation_vector = np.random.normal(0, 0.05 * np.std(offspring[cluster_indices], axis=0), self.dim)
                else:
                    mutation_vector = np.random.normal(0, 0.05 * np.std(offspring, axis=0), self.dim)
                offspring[i] = individual + mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            improvement = np.min([np.linalg.norm(solution - mem) for mem in self.memory])
            if improvement > 0.1:
                self.memory[np.argmin([np.linalg.norm(solution - mem) for mem in self.memory])] = solution

    def _adaptive_memory_management(self, population, fitness):
        if len(self.memory) > 0:
            mean_dist = np.mean([np.min([np.linalg.norm(ind - mem) for mem in self.memory]) for ind in population])
            self.mutation_rate = 0.1 if mean_dist > 0.5 else 0.2

    def _adaptive_parameter_adjustment(self, population, fitness):
        dist_matrix = dist.squareform(dist.pdist(population))
        clustering = AgglomerativeClustering(n_clusters=min(5, len(population) // 2), affinity='precomputed', linkage='average')
        clusters = clustering.fit_predict(dist_matrix)

        diversity_measure = len(set(clusters))
        if diversity_measure < self.population_size // 10:  # less diversity
            self.mutation_rate = min(self.mutation_rate * 1.1, 0.5)  # Adjusted
            self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)
        else:
            self.mutation_rate = max(self.mutation_rate * 0.9, 0.05)  # Adjusted
            self.crossover_rate = min(self.crossover_rate * 1.1, 0.9)

    def _adaptive_local_search(self, solution, func, bounds):
        step_size = 0.002
        for _ in range(20):
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            if func(candidate) < func(solution):
                solution = candidate
                step_size *= 1.2
            else:
                step_size *= 0.8
        return solution

    def _dimensionality_reduction(self, population):
        if self.dim > 10:
            pca = PCA(n_components=10)  # Reduce to 10 dimensions or fewer
            reduced_population = pca.fit_transform(population)
            restored_population = pca.inverse_transform(reduced_population)
            return restored_population
        return population