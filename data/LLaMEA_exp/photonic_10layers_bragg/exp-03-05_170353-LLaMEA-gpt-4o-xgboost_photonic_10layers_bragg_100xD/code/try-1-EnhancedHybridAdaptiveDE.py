import numpy as np
from scipy.spatial.distance import cdist

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.dynamic_niches_ratio = 0.1  # Fraction of population for niching
        self.chaos_factor = 0.7  # Initial chaos factor for parameter adaptation

    def chaotic_map(self):
        # Logistic map for chaos-inspired parameter adaptation
        self.chaos_factor = 4.0 * self.chaos_factor * (1 - self.chaos_factor)
        self.F = 0.4 + self.chaos_factor * 0.1
        self.CR = 0.8 + self.chaos_factor * 0.1

    def differential_evolution(self, func, bounds, population):
        trial_population = np.copy(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])
        return trial_population

    def dynamic_clustering(self, population, fitness):
        # Improved niching with dynamic clustering
        distances = cdist(population, population)
        cluster_indices = np.argmin(distances + np.eye(len(population)) * np.inf, axis=0)
        niche_count = int(self.dynamic_niches_ratio * self.population_size)
        unique_clusters = np.unique(cluster_indices)
        clusters = [population[cluster_indices == idx] for idx in unique_clusters]
        niches = [cluster[np.argmin(np.apply_along_axis(func, 1, cluster))] for cluster in clusters[:niche_count]]
        return niches

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.chaotic_map()  # Update F and CR using chaotic map
            trial_population = self.differential_evolution(func, bounds, population)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            # Select based on fitness
            better_idx = trial_fitness < fitness
            population[better_idx] = trial_population[better_idx]
            fitness[better_idx] = trial_fitness[better_idx]

            # Apply dynamic clustering
            niches = self.dynamic_clustering(population, fitness)
            for i in range(len(population)):
                if not any(np.array_equal(population[i], niche) for niche in niches):
                    population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]