import numpy as np
from sklearn.cluster import KMeans

class DualPopulationAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.main_population_size = 50
        self.secondary_population_size = int(0.5 * self.main_population_size)
        self.F = 0.5
        self.CR = 0.9
        self.main_population = None
        self.secondary_population = None
        self.best_agent = None
        self.evaluations = 0
        self.entropy_threshold = 0.1
        self.elite_fraction = 0.2
        self.covariance_matrix = np.identity(dim)
        self.mean_vector = np.zeros(dim)

    def initialize_populations(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.main_population = np.random.uniform(lb, ub, (self.main_population_size, self.dim))
        self.secondary_population = np.random.uniform(lb, ub, (self.secondary_population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.main_population])
        self.best_agent = self.main_population[np.argmin(fitness)]
        self.evaluations += self.main_population_size + self.secondary_population_size

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.F *= 1.03
            self.CR *= 0.95
        else:
            self.F *= 0.97
            self.CR *= 1.05

    def covariance_matrix_adaptation(self, fitness):
        improvement = np.mean(fitness) - np.min(fitness)
        if improvement > 0:
            self.covariance_matrix *= 1.05
        else:
            self.covariance_matrix *= 0.95

    def differential_evolution(self, target_idx, func, population, fitness):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(len(population)) if i != target_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, population[target_idx])
        trial_fitness = func(trial)
        if trial_fitness < fitness[target_idx]:
            population[target_idx] = trial
            fitness[target_idx] = trial_fitness
            if trial_fitness < func(self.best_agent):
                self.best_agent = trial

    def cluster_and_relocate(self, func, population, fitness):
        n_clusters = max(2, len(population) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(population)
        for cluster_idx in range(n_clusters):
            cluster_points = population[kmeans.labels_ == cluster_idx]
            cluster_center = np.mean(cluster_points, axis=0)
            if len(cluster_points) > 0:
                random_point = cluster_points[np.random.randint(len(cluster_points))]
                new_point = 0.5 * (cluster_center + random_point)
                new_point = np.clip(new_point, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_point)
                self.evaluations += 1
                if new_fitness < np.min(fitness[kmeans.labels_ == cluster_idx]):
                    min_idx = np.argmin(fitness[kmeans.labels_ == cluster_idx])
                    population[kmeans.labels_ == cluster_idx][min_idx] = new_point
                    fitness[kmeans.labels_ == cluster_idx][min_idx] = new_fitness
                    if new_fitness < func(self.best_agent):
                        self.best_agent = new_point

    def adaptive_gaussian_perturbation(self, agent, func):
        sigma = np.std(self.main_population, axis=0)
        g_noise = np.random.normal(0, sigma, self.dim)
        new_agent = np.clip(agent + g_noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def __call__(self, func):
        self.initialize_populations(func)
        main_fitness = np.array([func(ind) for ind in self.main_population])
        secondary_fitness = np.array([func(ind) for ind in self.secondary_population])

        while self.evaluations < self.budget:
            self.entropy_based_adaptation(main_fitness)
            self.covariance_matrix_adaptation(main_fitness)
            self.cluster_and_relocate(func, self.main_population, main_fitness)

            for idx in range(self.main_population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, self.main_population, main_fitness)
                self.evaluations += 1

            if self.evaluations < self.budget:
                for idx in range(self.secondary_population_size):
                    if self.evaluations >= self.budget:
                        break
                    candidate = self.adaptive_gaussian_perturbation(self.secondary_population[idx], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < secondary_fitness[idx]:
                        self.secondary_population[idx] = candidate
                        secondary_fitness[idx] = candidate_fitness
                        self.evaluations += 1
                        if candidate_fitness < func(self.best_agent):
                            self.best_agent = candidate

        return self.best_agent