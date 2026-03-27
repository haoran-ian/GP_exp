import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class EnhancedFuzzyChaosDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5
        self.CR_base = 0.9
        self.fuzzy_clustering_ratio = 0.15
        self.epsilon = 1e-3
        self.chaos_factor = 0.1

    def adaptive_parameters(self, generation):
        chaotic_sequence = np.sin(generation * self.chaos_factor)
        F = self.F_base + chaotic_sequence * np.random.rand()
        CR = self.CR_base * chaotic_sequence
        return np.clip(F, 0.3, 0.8), np.clip(CR, 0.1, 1.0)

    def differential_evolution(self, func, bounds, population, generation):
        trial_population = np.copy(population)
        F, CR = self.adaptive_parameters(generation)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])
        return trial_population

    def fuzzy_clustering(self, population, fitness):
        niche_count = int(self.fuzzy_clustering_ratio * self.population_size)
        kmeans = KMeans(n_clusters=max(2, niche_count), random_state=0).fit(population)
        niches = kmeans.cluster_centers_
        return niches

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            trial_population = self.differential_evolution(func, bounds, population, generation)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            better_idx = trial_fitness + self.epsilon < fitness
            population[better_idx] = trial_population[better_idx]
            fitness[better_idx] = trial_fitness[better_idx]

            niches = self.fuzzy_clustering(population, fitness)
            distance_matrix = cdist(niches, population)
            for i in range(len(population)):
                if np.any(distance_matrix[:, i] < 0.1):
                    population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]