import numpy as np

class HybridDE_SA_TD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.98

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            scaling_factor = 0.5 + 0.4 * (iteration / self.max_iterations)  # Scaling factor changes over time
            population_size = self.initial_population_size - int(iteration * (self.initial_population_size / self.max_iterations))  # Dynamic population size
            adaptive_rate = 0.9 - 0.6 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            for i in range(population_size):
                indices = np.random.permutation(population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation with time-varying scaling factor
                mutant = x1 + scaling_factor * (x2 - x3)
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