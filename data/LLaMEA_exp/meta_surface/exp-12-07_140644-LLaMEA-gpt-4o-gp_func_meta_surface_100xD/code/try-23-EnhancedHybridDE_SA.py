import numpy as np

class EnhancedHybridDE_SA:
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
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            adaptive_mutation_factor = 0.5 + 0.5 * (iteration / self.max_iterations)  # Adaptive mutation strength
            dynamic_population_factor = 1.0 - (iteration / self.max_iterations)  # Dynamic population size adjustment

            # Dynamically adjust population size
            self.population_size = int(self.initial_population_size * dynamic_population_factor) or 1

            for i in range(self.population_size):
                indices = np.random.permutation(self.initial_population_size)[:3]
                x1, x2, x3 = population[indices]

                # Differential Evolution Mutation with Adaptive Mutation Factor
                mutant = x1 + adaptive_mutation_factor * (x2 - x3)
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

        return best_solution