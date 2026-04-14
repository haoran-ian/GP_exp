import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cluster_count = 5
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.memory_size = 20
        self.memory = []
        self.evaluations = 0
        self.local_search_intensity = 0.002
        self.scale_intensity = 1.0

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
            
            population = self._cluster_based_selection(population, fitness)
            offspring = self._crossover(population, bounds)
            population = self._mutation(offspring, bounds)

            best_solution = self._adaptive_local_search(best_solution, func, bounds)
            self._dynamic_exploration_exploitation_balance(population, fitness)

        return best_solution

    def _initialize_population(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))

    def _cluster_based_selection(self, population, fitness):
        scaler = MinMaxScaler()
        scaled_fitness = scaler.fit_transform(fitness.reshape(-1, 1)).flatten()
        kmeans = KMeans(n_clusters=self.cluster_count, n_init=10)
        clusters = kmeans.fit_predict(population)

        selected_population = []
        for cluster in range(self.cluster_count):
            cluster_indices = np.where(clusters == cluster)[0]
            if len(cluster_indices) > 0:
                best_idx = cluster_indices[np.argmin(scaled_fitness[cluster_indices])]
                selected_population.append(population[best_idx])
        
        return np.array(selected_population)

    def _crossover(self, population, bounds):
        offspring = []
        for i in range(len(population)):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
                alpha = np.random.rand(self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2
                offspring.append(child)
            else:
                offspring.append(population[i])
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def _mutation(self, offspring, bounds):
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, self.scale_intensity * 0.05 * np.std(offspring, axis=0), self.dim)
                offspring[i] += mutation_vector
                offspring[i] = np.clip(offspring[i], bounds[:, 0], bounds[:, 1])
        return offspring

    def _update_memory(self, solution):
        if len(self.memory) < self.memory_size:
            self.memory.append(solution)
        else:
            improvement = np.min([np.linalg.norm(solution - mem) for mem in self.memory])
            if improvement > 0.1:
                self.memory[np.argmin([np.linalg.norm(solution - mem) for mem in self.memory])] = solution

    def _dynamic_exploration_exploitation_balance(self, population, fitness):
        diversity = np.mean(dist.pdist(population))
        if diversity < 0.1:
            self.crossover_rate *= 0.9
            self.mutation_rate *= 1.1
            self.scale_intensity *= 1.1
        else:
            self.crossover_rate *= 1.1
            self.mutation_rate *= 0.9
            self.scale_intensity *= 0.9

    def _adaptive_local_search(self, solution, func, bounds):
        step_size = self.local_search_intensity
        for _ in range(30):
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            if func(candidate) < func(solution):
                solution = candidate
                step_size *= 1.3
            else:
                step_size *= 0.7
        return solution