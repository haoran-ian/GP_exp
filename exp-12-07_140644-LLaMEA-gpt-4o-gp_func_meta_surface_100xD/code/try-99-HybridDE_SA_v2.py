import numpy as np

class HybridDE_SA_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            mutation_scaling = 0.6 + 0.3 * (np.cos(np.pi * iteration / self.max_iterations))  # Adaptive mutation scaling
            adaptive_rate = 0.9 - 0.6 * (iteration / self.max_iterations)

            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation with adaptive scaling
                mutant = x1 + mutation_scaling * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            # Cooling
            self.temperature *= self.cooling_rate

            # Dynamic population resizing
            if iteration % (self.max_iterations // 5) == 0 and self.population_size > self.initial_population_size // 2:
                self.population_size = max(self.population_size // 2, self.initial_population_size // 2)
                new_population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                new_fitness = np.array([func(ind) for ind in new_population])
                combined_population = np.vstack((population, new_population))
                combined_fitness = np.hstack((fitness, new_fitness))
                sorted_indices = np.argsort(combined_fitness)[:self.population_size]
                population = combined_population[sorted_indices]
                fitness = combined_fitness[sorted_indices]

        return best_solution