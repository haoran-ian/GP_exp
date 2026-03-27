import numpy as np
from scipy.spatial.distance import cdist

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR = 0.9  # Crossover probability
        self.dynamic_niches_ratio = 0.1  # Fraction of population for niching
        self.elite_fraction = 0.1  # Fraction of population considered as elite
        self.crowding_radius = 0.1 

    def differential_evolution(self, func, bounds, population, fitness):
        trial_population = np.copy(population)
        elite_size = max(1, int(self.elite_fraction * self.population_size))
        elite_indices = np.argsort(fitness)[:elite_size]
        elites = population[elite_indices]

        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            F = self.F_base + np.random.rand() * (1 - self.F_base)  # Adaptive mutation factor
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])

        # Fitness-based crowding
        for i in range(self.population_size):
            distances = cdist([trial_population[i]], elites)
            if np.any(distances < self.crowding_radius):
                trial_population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
        
        return trial_population

    def dynamic_niching(self, population, fitness):
        niches = []
        niche_count = int(self.dynamic_niches_ratio * self.population_size)

        while len(niches) < niche_count:
            idx = np.random.choice(len(population))
            if not any(np.allclose(population[idx], niche, atol=1e-1) for niche in niches):
                niches.append(population[idx])
                
        return niches

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            trial_population = self.differential_evolution(func, bounds, population, fitness)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            # Select based on fitness
            better_idx = trial_fitness < fitness
            population[better_idx] = trial_population[better_idx]
            fitness[better_idx] = trial_fitness[better_idx]

            # Apply dynamic niching
            niches = self.dynamic_niching(population, fitness)
            distance_matrix = cdist(niches, population)
            for i in range(len(population)):
                if np.any(distance_matrix[:, i] < 1e-1):
                    population[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    fitness[i] = func(population[i])
                    evaluations += 1

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]