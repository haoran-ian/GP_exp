import numpy as np
from scipy.spatial.distance import cdist

class ChaosEnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.dynamic_niches_ratio = 0.1  # Fraction of population for niching
        self.epsilon = 1e-3  # Small value to prevent division by zero
        self.chaos_factor = 0.1  # Chaotic influence factor

    def adaptive_parameters(self, generation):
        chaotic_sequence = np.tan(generation * self.chaos_factor)
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

    def local_search(self, func, individual):
        grad_step = 1e-3
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            perturbed = np.copy(individual)
            perturbed[d] += grad_step
            gradient[d] = (func(perturbed) - func(individual)) / grad_step
        return individual - 0.02 * gradient

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
            trial_population = self.differential_evolution(func, bounds, population, generation)
            trial_fitness = np.apply_along_axis(func, 1, trial_population)
            evaluations += self.population_size

            for i in range(self.population_size):
                if trial_fitness[i] + self.epsilon < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = trial_fitness[i]
                else:  # Apply local search on worse solutions
                    population[i] = self.local_search(func, population[i])
                    fitness[i] = func(population[i])
                    evaluations += 1

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