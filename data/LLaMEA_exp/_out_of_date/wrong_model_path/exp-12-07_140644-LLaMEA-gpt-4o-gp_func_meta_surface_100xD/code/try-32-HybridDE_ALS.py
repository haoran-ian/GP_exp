import numpy as np

class HybridDE_ALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.initial_temp = 100.0
        self.cooling_rate = 0.98
        self.elite_ratio = 0.1
        self.mutation_factor_base = 0.8

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            temperature = self.initial_temp * (self.cooling_rate ** iteration)
            elite_count = max(1, int(self.elite_ratio * self.population_size))
            sorted_indices = np.argsort(fitness)
            elite_indices = sorted_indices[:elite_count]
            adaptive_mutation_factor = self.mutation_factor_base * (1 - iteration / self.max_iterations)

            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2 = population[indices[:2]]

                # Elite Guided Perturbation: Combine with elite solutions
                elite_idx = np.random.choice(elite_indices)
                mutant = x1 + adaptive_mutation_factor * (x2 - population[elite_idx])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover_mask, mutant, population[i])

                # Adaptive Local Search Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

        return best_solution