import numpy as np
from scipy.spatial.distance import cdist

class EnhancedChaosSynergyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.dynamic_niches_ratio = 0.1  # Fraction of population for niching
        self.epsilon = 1e-3  # Small value to prevent division by zero
        self.chaos_factor = 0.1  # Chaotic influence factor
        self.collaboration_factor = 0.3  # Factor for collaborative crossover

        # Initialize multiple sub-populations for improved diversity
        self.sub_pop_count = 3
        self.sub_populations = [np.random.uniform(0, 1, (self.population_size, self.dim)) for _ in range(self.sub_pop_count)]

    def adaptive_parameters(self, generation):
        # Adapt F and CR using chaotic maps
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

        # Collaborative crossover
        if generation % 5 == 0:
            other_pop = self.sub_populations[(generation // 5) % self.sub_pop_count]
            for i in range(self.population_size):
                collaborator = other_pop[np.random.randint(self.population_size)]
                trial_population[i] = (1 - self.collaboration_factor) * trial_population[i] + self.collaboration_factor * collaborator

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
        fitness_records = []
        for pop_idx in range(self.sub_pop_count):
            population = self.sub_populations[pop_idx] * (bounds.ub - bounds.lb) + bounds.lb
            fitness = np.apply_along_axis(func, 1, population)
            evaluations = self.population_size
            generation = 0

            while evaluations < self.budget / self.sub_pop_count:
                trial_population = self.differential_evolution(func, bounds, population, generation)
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

            fitness_records.append((population, fitness))

        # Merge results from all sub-populations and return the best solution
        all_populations = np.vstack([record[0] for record in fitness_records])
        all_fitness = np.hstack([record[1] for record in fitness_records])
        best_idx = np.argmin(all_fitness)
        return all_populations[best_idx], all_fitness[best_idx]