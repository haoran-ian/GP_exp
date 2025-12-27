import numpy as np
from sklearn.cluster import KMeans

class QuantumClusteredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.population = None
        self.best_agent = None
        self.evaluations = 0
        self.entropy_threshold = 0.1
        self.quantum_factor = 0.1 
        self.elite_fraction = 0.2 

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.quantum_factor *= 1.05
        else:
            self.quantum_factor *= 0.95

    def quantum_inspired_search(self, agent, func):
        z = np.random.normal(0, self.quantum_factor, self.dim)
        new_agent = np.clip(agent + z, func.bounds.lb, func.bounds.ub)
        return new_agent

    def adaptive_clustering(self, func, fitness):
        n_clusters = max(2, self.population_size // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.population)
        for cluster_idx in range(n_clusters):
            cluster_points = self.population[kmeans.labels_ == cluster_idx]
            cluster_center = np.mean(cluster_points, axis=0)
            random_point = cluster_points[np.random.randint(len(cluster_points))]
            new_point = 0.5 * (cluster_center + random_point)
            new_point = np.clip(new_point, func.bounds.lb, func.bounds.ub)
            new_fitness = func(new_point)
            self.evaluations += 1
            if new_fitness < np.min(fitness[kmeans.labels_ == cluster_idx]):
                min_idx = np.argmin(fitness[kmeans.labels_ == cluster_idx])
                self.population[kmeans.labels_ == cluster_idx][min_idx] = new_point
                fitness[kmeans.labels_ == cluster_idx][min_idx] = new_fitness
                if new_fitness < func(self.best_agent):
                    self.best_agent = new_point

    def elite_preservation(self, fitness):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness)[:elite_count]
        return self.population[elite_indices]

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.entropy_based_adaptation(fitness)
            self.adaptive_clustering(func, fitness)
            elite_population = self.elite_preservation(fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                candidate = self.quantum_inspired_search(self.population[idx], func)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[idx]:
                    self.population[idx] = candidate
                    fitness[idx] = candidate_fitness
                    self.evaluations += 1
                    if candidate_fitness < func(self.best_agent):
                        self.best_agent = candidate

        return self.best_agent