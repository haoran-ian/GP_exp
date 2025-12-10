import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.min_population_size = 4 * dim
        self.max_iterations = 1000  # Set a reasonable upper limit for iterations
        self.temperature = 100.0
        self.cooling_rate = 0.995

    def __call__(self, func):
        remaining_budget = self.budget
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        iteration = 0
        while remaining_budget > 0 and iteration < self.max_iterations:
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            for i in range(population_size):
                indices = np.random.permutation(population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation
                mutant = x1 + 0.8 * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                remaining_budget -= 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if remaining_budget <= 0:
                    break

            # Cooling
            self.temperature *= self.cooling_rate

            # Adaptive Population Resizing
            if iteration % 100 == 0 and population_size > self.min_population_size:
                population_size = max(self.min_population_size, int(population_size * 0.9))
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices[:population_size]]
                fitness = fitness[sorted_indices[:population_size]]

            iteration += 1

        return best_solution