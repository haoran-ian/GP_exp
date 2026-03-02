import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveMemoryChaoticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        exploration_factor = 0.6
        exploitation_factor = 0.4
        initial_mutation_rate = 0.15
        mutation_rate = initial_mutation_rate

        while self.evaluations < self.budget:
            clusters, centroids = self._dynamic_clustering(population, population_size // 4)
            memory_impression = self._calculate_memory_impression(fitness)
            exploration_weight = exploration_factor * (1 - memory_impression)
            exploitation_weight = exploitation_factor * memory_impression

            for i, individual in enumerate(population):
                trial, elitism_factor = self._adaptive_perturbation(individual, lb, ub, mutation_rate, memory_impression, clusters, centroids, i)
                trial_fitness = self._evaluate(func, trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                self.memory.append((trial, trial_fitness))

            for i, individual in enumerate(population):
                if np.random.rand() < exploitation_weight:
                    trial = self._enhanced_chaos_local_search(population, fitness, i, lb, ub, func, centroids)
                    trial_fitness = self._evaluate(func, trial)
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness

            mutation_rate = self._adapt_mutation_rate(memory_impression, initial_mutation_rate, fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _adaptive_perturbation(self, individual, lb, ub, mutation_rate, memory_impression, clusters, centroids, index):
        cluster_id = clusters[index]
        cluster_center = centroids[cluster_id]
        if self.memory:
            best_memory_ind = min(self.memory, key=lambda x: x[1])[0]
        else:
            best_memory_ind = individual
        chaos_factor = np.random.normal(0, mutation_rate * 1.5, size=self.dim)
        perturbation = np.sin(chaos_factor) * mutation_rate
        elitism_factor = 0.3 + 0.3 * memory_impression
        cluster_adjustment = 0.5 * (cluster_center - individual) * (1 - memory_impression)
        trial = np.clip(best_memory_ind + perturbation * elitism_factor + cluster_adjustment, lb, ub)
        return trial, elitism_factor

    def _enhanced_chaos_local_search(self, population, fitness, index, lb, ub, func, centroids):
        neighbors = self._get_neighbors(population, index)
        best_neighbor = min(neighbors, key=lambda ind: func(ind))
        weighted_direction = 0.7 * (best_neighbor - population[index])
        chaos_direction = np.sin(weighted_direction) * 0.25
        cluster_center = centroids[np.random.choice(range(len(centroids)))]
        cluster_direction = 0.5 * (cluster_center - population[index])
        chaos_cluster_adj = np.sin(cluster_direction) * 0.1
        trial = np.clip(population[index] + chaos_direction + chaos_cluster_adj, lb, ub)
        return trial

    def _get_neighbors(self, population, index):
        neighbor_indices = np.random.choice(len(population), min(3, len(population)-1), replace=False)
        neighbors = population[neighbor_indices]
        return neighbors

    def _calculate_memory_impression(self, fitness):
        if not self.memory:
            return 0
        memory_fitness = np.array([fit for _, fit in self.memory])
        global_best_memory = np.min(memory_fitness)
        global_worst_memory = np.max(memory_fitness)
        return (global_best_memory - np.mean(memory_fitness)) / (global_worst_memory - global_best_memory + 1e-6)

    def _adapt_mutation_rate(self, memory_impression, initial_mutation_rate, fitness):
        fitness_variance = np.var(fitness)
        return initial_mutation_rate * (1 + 0.5 * memory_impression) * (1 + fitness_variance)

    def _dynamic_clustering(self, population, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters).fit(population)
        return kmeans.labels_, kmeans.cluster_centers_