import numpy as np
from sklearn.cluster import KMeans

class EnhancedClusteredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_agent = None
        self.evaluations = 0
        self.stagnation_threshold = 0.1 * self.budget
        self.last_improvement_evaluation = 0
        self.entropy_threshold = 0.1
        self.covariance_matrix = np.identity(dim)
        self.mean_vector = np.zeros(dim)
        self.elite_fraction = 0.2

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.last_improvement_evaluation = self.evaluations
        self.evaluations += self.population_size

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.F *= 1.05
            self.CR *= 0.9
        else:
            self.F *= 0.95
            self.CR *= 1.1

    def covariance_matrix_adaptation(self, fitness):
        improvement = np.mean(fitness) - np.min(fitness)
        if improvement > 0:
            self.covariance_matrix *= 1.1
        else:
            self.covariance_matrix *= 0.9

    def hybrid_clustering(self, func, fitness):
        n_clusters = max(2, self.population_size // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.population)
        for cluster_idx in range(n_clusters):
            cluster_points = self.population[kmeans.labels_ == cluster_idx]
            if len(cluster_points) > 0:
                cluster_fitness = fitness[kmeans.labels_ == cluster_idx]
                best_idx = np.argmin(cluster_fitness)
                cluster_center = cluster_points[best_idx]
                perturbation = np.clip(np.random.normal(0, 1, self.dim), func.bounds.lb, func.bounds.ub)
                new_point = np.clip(cluster_center + perturbation, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_point)
                self.evaluations += 1
                if new_fitness < cluster_fitness[best_idx]:
                    self.population[kmeans.labels_ == cluster_idx][best_idx] = new_point
                    fitness[kmeans.labels_ == cluster_idx][best_idx] = new_fitness
                    if new_fitness < func(self.best_agent):
                        self.best_agent = new_point
                        self.last_improvement_evaluation = self.evaluations

    def elite_preservation(self, fitness):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness)[:elite_count]
        return self.population[elite_indices]

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
                self.last_improvement_evaluation = self.evaluations

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.entropy_based_adaptation(fitness)
            self.covariance_matrix_adaptation(fitness)
            self.hybrid_clustering(func, fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1

        return self.best_agent