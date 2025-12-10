import numpy as np

class AugmentedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.population_growth_factor = 1.1  # Factor to increase population size

    def __call__(self, func):
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.6 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            for i in range(population_size):
                indices = np.random.permutation(population_size)
                x1, x2, x3 = population[indices[:3]]

                # Dynamic scaling of differential weight
                F = 0.5 + 0.5 * np.random.rand()
                mutant = x1 + F * (x2 - x3)
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

            # Dynamic population scaling
            if iteration % 10 == 0 and iteration > 0:  # Increase population size every 10 iterations
                population_size = min(int(population_size * self.population_growth_factor), int(self.budget / self.max_iterations))
                additional_population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size - len(population), self.dim))
                population = np.vstack((population, additional_population))
                additional_fitness = np.array([func(ind) for ind in additional_population])
                fitness = np.hstack((fitness, additional_fitness))
                
        return best_solution