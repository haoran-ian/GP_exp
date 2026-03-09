import numpy as np
from scipy.spatial.distance import cdist

class EnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5
        self.CR_base = 0.9
        self.dynamic_niches_ratio = 0.1
        self.epsilon = 1e-3
        self.diversity_threshold = 0.1  # Threshold for diversity adaptation

    def adaptive_parameters(self, generation, diversity):
        # Adapt F and CR based on generation and diversity
        F = self.F_base + 0.5 * np.random.rand() * (1 + diversity)
        CR = self.CR_base * np.random.rand() * (1 - diversity)
        return F, CR

    def calculate_diversity(self, population):
        # Measure population diversity as the average distance between individuals
        distances = cdist(population, population)
        mean_distance = np.mean(distances)
        max_distance = np.sqrt(self.dim)  # Max possible distance in normalized space
        return mean_distance / max_distance

    def differential_evolution(self, func, bounds, population, generation, diversity):
        trial_population = np.copy(population)
        F, CR = self.adaptive_parameters(generation, diversity)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_population[i] = np.where(cross_points, mutant, population[i])
        return trial_population

    def dynamic_niching(self, population, fitness):
        niches = []
        niche_count = int(self.dynamic_niches_ratio * self.population_size)
        while len(niches) < niche_count:
            idx = np.random.choice(len(population))
            if not any(np.allclose(population[idx], niche, atol=0.1) for niche in niches):
                niches.append(population[idx])
        return niches

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            diversity = self.calculate_diversity(population)
            trial_population = self.differential_evolution(func, bounds, population, generation, diversity)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            # Select based on fitness
            better_idx = trial_fitness + self.epsilon < fitness
            population[better_idx] = trial_population[better_idx]
            fitness[better_idx] = trial_fitness[better_idx]

            # Apply dynamic niching
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