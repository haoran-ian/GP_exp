import numpy as np

class HybridDE_SA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + dim  # Dynamic population size
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.97  # Slightly adjusted cooling rate
        self.mutation_factor = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation factor

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < 0.9
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

            # Adaptive mutation factor adjustment
            self.mutation_factor = 0.5 + 0.3 * np.random.rand()

            # Cooling
            self.temperature *= self.cooling_rate

        return best_solution