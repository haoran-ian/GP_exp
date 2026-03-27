import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.initial_temperature = 100.0
        self.cooling_rate = 0.98  # Slightly more aggressive cooling
        self.cr_min = 0.5
        self.cr_max = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]

                # Adaptive Differential Evolution Mutation
                F = 0.5 + (0.5 * np.random.rand())  # Adaptive mutation factor
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Dynamic Crossover Rate
                cr = self.cr_min + (self.cr_max - self.cr_min) * (best_fitness / (fitness[i] + 1e-8))
                crossover_mask = np.random.rand(self.dim) < cr
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness

            # Cooling
            temperature *= self.cooling_rate

        return best_solution