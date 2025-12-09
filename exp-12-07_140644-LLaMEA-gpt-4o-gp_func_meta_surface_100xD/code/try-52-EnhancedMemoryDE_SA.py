import numpy as np

class EnhancedMemoryDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.elite_size = max(1, self.population_size // 10)  # Elite archive size
        self.elite_archive = []

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.5 * (iteration / self.max_iterations)
            f_scaling = 0.5 + 0.3 * (np.sin(iteration / self.max_iterations * np.pi))  # Adaptive scaling factor
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]
                
                # Differential Evolution Mutation with elite consideration
                if np.random.rand() < 0.1 and self.elite_archive:
                    x1 = self.elite_archive[np.random.randint(len(self.elite_archive))]
                
                mutant = x1 + f_scaling * (x2 - x3)
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

            # Update elite archive
            elite_indices = np.argsort(fitness)[:self.elite_size]
            self.elite_archive = [population[idx] for idx in elite_indices if fitness[idx] < best_fitness]

            # Cooling
            self.temperature *= self.cooling_rate

        return best_solution