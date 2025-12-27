import numpy as np
from sklearn.cluster import KMeans

class RefinedClusteredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.min_population_size = 10
        self.F_base = 0.5
        self.CR_base = 0.9
        self.mutation_scale = 1.0
        self.population = None
        self.best_agent = None
        self.evaluations = 0
        self.entropy_threshold = 0.1
        self.elite_fraction = 0.2
        self.covariance_matrix = np.identity(dim)

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size

    def adapt_mutation_scale(self, fitness):
        fitness_diff = np.max(fitness) - np.min(fitness)
        self.mutation_scale = max(0.1, 1.0 * (fitness_diff / max(1e-9, np.sum(fitness))))

    def fitness_weighted_migration(self, func, fitness):
        weights = 1.0 / (1.0 + fitness - np.min(fitness))
        weighted_population = self.population * weights[:, np.newaxis]
        new_population = np.random.permutation(weighted_population)
        
        for idx in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            candidate = new_population[idx]
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < fitness[idx]:
                self.population[idx] = candidate
                fitness[idx] = candidate_fitness
                if candidate_fitness < func(self.best_agent):
                    self.best_agent = candidate

    def differential_evolution(self, target_idx, func, fitness):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_scale * (b - c), lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR_base
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        trial_fitness = func(trial)
        if trial_fitness < fitness[target_idx]:
            self.population[target_idx] = trial
            fitness[target_idx] = trial_fitness
            if trial_fitness < func(self.best_agent):
                self.best_agent = trial

    def adaptive_clustering(self, func, fitness):
        n_clusters = max(2, self.population_size // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.population)
        for cluster_idx in range(n_clusters):
            cluster_points = self.population[kmeans.labels_ == cluster_idx]
            if len(cluster_points) > 0:
                random_point = cluster_points[np.random.randint(len(cluster_points))]
                cluster_center = np.mean(cluster_points, axis=0)
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
            self.adapt_mutation_scale(fitness)
            self.fitness_weighted_migration(func, fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1
            self.adaptive_clustering(func, fitness)
            self.elite_preservation(fitness)

        return self.best_agent