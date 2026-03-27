import numpy as np

class AdaptiveHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.dynamic_population_size = self.initial_population_size
        self.max_iterations = budget // self.initial_population_size
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.elite_portion = 0.2

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)  # Dynamically adjusted crossover rate

            # Dynamic population resizing
            self.dynamic_population_size = max(self.initial_population_size // 2, int(self.initial_population_size * (1 - iteration / self.max_iterations)))
            elite_size = max(2, int(self.dynamic_population_size * self.elite_portion))
            
            # Select elites
            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]

            for i in range(self.dynamic_population_size):
                indices = np.random.permutation(elite_size)
                x1, x2, x3 = elites[indices[:3]]

                # Differential Evolution Mutation
                mutant = x1 + 0.8 * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < adaptive_rate
                trial = np.where(crossover_mask, mutant, population[i % self.initial_population_size])

                # Simulated Annealing Acceptance
                trial_fitness = func(trial)
                if trial_fitness < fitness[i % self.initial_population_size] or np.random.rand() < np.exp((fitness[i % self.initial_population_size] - trial_fitness) / self.temperature):
                    population[i % self.initial_population_size] = trial
                    fitness[i % self.initial_population_size] = trial_fitness

                # Update best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            # Cooling
            self.temperature *= self.cooling_rate

        return best_solution