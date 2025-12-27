import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm

class MultiPhaseAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.9
        self.evaluations = 0
        self.population = None
        self.best_agent = None
        self.elite_fraction = 0.2
        self.cov_matrix = np.identity(dim)
        self.mean_vector = np.zeros(dim)
        self.phase_switch_threshold = budget * 0.5

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size

    def differential_evolution(self, target_idx, func, fitness):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.population[target_idx])
        trial_fitness = func(trial)
        if trial_fitness < fitness[target_idx]:
            self.population[target_idx] = trial
            fitness[target_idx] = trial_fitness
            if trial_fitness < func(self.best_agent):
                self.best_agent = trial

    def elite_preservation(self, fitness):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness)[:elite_count]
        return self.population[elite_indices]

    def adaptive_clustering(self, func, fitness):
        if self.evaluations < self.phase_switch_threshold:
            n_clusters = max(2, self.population_size // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.population)
            for cluster_idx in range(n_clusters):
                cluster_points = self.population[kmeans.labels_ == cluster_idx]
                if len(cluster_points) > 0:
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

    def covariance_guided_search(self, func):
        if self.evaluations >= self.phase_switch_threshold:
            for idx in range(self.population_size):
                z = np.random.multivariate_normal(self.mean_vector, self.cov_matrix)
                candidate = np.clip(self.population[idx] + z, func.bounds.lb, func.bounds.ub)
                candidate_fitness = func(candidate)
                self.evaluations += 1
                if candidate_fitness < func(self.best_agent):
                    self.best_agent = candidate

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])
        while self.evaluations < self.budget:
            self.adaptive_clustering(func, fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1
            self.covariance_guided_search(func)
        return self.best_agent