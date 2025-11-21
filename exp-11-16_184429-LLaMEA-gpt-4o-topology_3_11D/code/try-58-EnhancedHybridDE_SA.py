import numpy as np

class EnhancedHybridDE_SA:
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
            F = 0.5 + 0.3 * (1 - self.current_budget / self.budget)  # Adaptive mutation factor
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x1 - x2), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / (1 + self.current_budget / self.budget)):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

            # Elitism: Preserve the best solution found
            elite_idx = np.argmin(fitness)
            if fitness[elite_idx] < best_fitness:
                best = population[elite_idx]
                best_fitness = fitness[elite_idx]

        return best

# Example usage:
# optimizer = EnhancedHybridDE_SA(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)