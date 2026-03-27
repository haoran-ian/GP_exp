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
        self.final_temperature = 1.0
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            current_temperature = self.initial_temperature * ((self.final_temperature / self.initial_temperature) ** (iteration / self.max_iterations))
            adaptive_mutation_factor = self.mutation_factor + 0.2 * (1 - iteration / self.max_iterations)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # Differential Evolution Mutation with Adaptive Mutation Factor
                mutant = x1 + adaptive_mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                delta_fitness = fitness[i] - trial_fitness
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(delta_fitness / current_temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

        return best_solution