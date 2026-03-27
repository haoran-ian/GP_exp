import numpy as np

class HybridDE_SA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.temperature = 1.0
        self.cooling_rate = 0.95
        self.scaling_factor = 0.8
        self.temperature_schedule = np.linspace(1.0, 0.01, self.max_iterations)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            self.scaling_factor = 0.8 - 0.5 * np.cos(np.pi * iteration / self.max_iterations)  # Adaptive scaling factor
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation with adaptive scaling factor
                mutant = x1 + self.scaling_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature_schedule[iteration]):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

        return best_solution