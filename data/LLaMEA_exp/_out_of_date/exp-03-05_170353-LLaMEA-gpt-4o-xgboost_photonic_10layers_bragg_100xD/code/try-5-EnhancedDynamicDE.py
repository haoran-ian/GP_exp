import numpy as np
from scipy.spatial.distance import cdist

class EnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.dynamic_niches_ratio = 0.1  # Fraction of population for niching
        self.epsilon = 1e-3  # Small value to prevent division by zero

    def adaptive_parameters(self, generation):
        # Adapt F and CR based on generations
        return self.F_base + 0.5 * np.random.rand(), self.CR_base * np.random.rand()

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

    def multi_scale_clustering(self, population):
        # Implement multi-scale clustering to identify niche centers
        dist_matrix = cdist(population, population)
        scale_steps = [0.1, 0.5, 1.0]
        niche_centers = set()

        for scale in scale_steps:
            for i, point in enumerate(population):
                neighbors = np.where(dist_matrix[i] < scale)[0]
                avg_point = np.mean(population[neighbors], axis=0)
                niche_centers.add(tuple(avg_point))

        return np.array(list(niche_centers))

    def dynamic_niching(self, population, fitness):
        niches = self.multi_scale_clustering(population)
        niche_count = int(self.dynamic_niches_ratio * self.population_size)
        selected_niches = np.random.choice(range(len(niches)), niche_count, replace=False)
        return niches[selected_niches]

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

            # Select based on fitness
            better_idx = trial_fitness + self.epsilon < fitness
            population[better_idx] = trial_population[better_idx]
            fitness[better_idx] = trial_fitness[better_idx]

            # Apply dynamic niching with multi-scale clustering
            niches = self.dynamic_niching(population, fitness)
            distance_matrix = cdist(niches, population)
            for i in range(len(population)):
                if np.any(distance_matrix[:, i] < 0.1):
                    population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

            generation += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]