import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.min_population_size = 4
        self.population_size = self.initial_population_size

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        self.evaluate_budget = self.population_size

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.6 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Differential Evolution Mutation
                F = 0.5 + 0.3 * np.random.rand()  # Self-adaptive scaling factor
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                self.evaluate_budget += 1
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
            if self.population_size > self.min_population_size:
                self.population_size = max(self.min_population_size, int(self.population_size * 0.95))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

            if self.evaluate_budget >= self.budget:
                break

        return best_solution