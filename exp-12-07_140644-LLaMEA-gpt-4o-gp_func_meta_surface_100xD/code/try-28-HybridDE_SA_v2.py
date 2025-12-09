import numpy as np

class HybridDE_SA_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.min_population_size = 4 * dim  # adaptive population size

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)
            for i in range(population_size):
                indices = np.random.permutation(population_size)
                x1, x2, x3 = population[indices[:3]]

                mutant = x1 + 0.8 * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            self.temperature *= self.cooling_rate

            # Adaptive Population Size Reduction
            if iteration % (self.max_iterations // 3) == 0 and population_size > self.min_population_size:
                population_size = max(self.min_population_size, int(population_size * 0.8))
                population, fitness = self._reduce_population(population, fitness, population_size)

        return best_solution

    def _reduce_population(self, population, fitness, new_size):
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[:new_size]], fitness[sorted_indices[:new_size]]