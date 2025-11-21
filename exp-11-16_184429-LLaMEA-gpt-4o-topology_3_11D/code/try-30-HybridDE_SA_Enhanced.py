import numpy as np

class HybridDE_SA_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.current_budget = 0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_budget < self.budget:
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break

                # Adaptive Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F = 0.5 + 0.3 * np.random.rand()  # Adaptive scaling factor
                mutant = np.clip(x0 + F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                CR = 0.7 + 0.2 * np.random.rand()  # Adaptive crossover rate
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial
                trial_fitness = func(trial)
                self.current_budget += 1

                # Simulated Annealing-inspired acceptance with dynamic temperature
                temperature = 1 + (5 * (1 - self.current_budget / self.budget))
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

        return best