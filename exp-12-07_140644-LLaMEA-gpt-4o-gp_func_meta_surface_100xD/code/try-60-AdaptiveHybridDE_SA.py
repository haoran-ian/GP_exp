import numpy as np

class AdaptiveHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 5 * dim)  # Dynamic initial population size
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.95  # Adjusted cooling rate for longer exploration
        self.mutation_factor = 0.8

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 * (1 - (iteration / self.max_iterations)) + 0.1  # Dynamic crossover rate
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Adaptive Mutation Factor
                self.mutation_factor = 0.5 + 0.3 * np.random.rand()
                mutant = x1 + self.mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance with adaptive temperature impact
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                # Early stopping criteria based on fitness evaluations
                if evaluations >= self.budget:
                    return best_solution

            # Cooling
            self.temperature *= self.cooling_rate

            # Dynamic population scaling
            if iteration % 10 == 0 and self.population_size < 2 * self.dim:
                new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                evaluations += 5
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitness))
                self.population_size = population.shape[0]

        return best_solution