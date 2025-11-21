import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.current_budget = 0

    def __call__(self, func):
        np.random.seed(42)
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        initial_temperature = 1.0
        final_temperature = 0.01
        temperature_decay = (final_temperature / initial_temperature) ** (1.0 / (self.budget - 1))

        while self.current_budget < self.budget:
            adaptive_mutation_rate = 0.5 + 0.3 * np.random.rand()
            adaptive_crossover_rate = 0.7 + 0.2 * np.random.rand()
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + adaptive_mutation_rate * (x1 - x2), self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < adaptive_crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                self.current_budget += 1

                temperature = initial_temperature * (temperature_decay ** self.current_budget)
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

        return best

# Example usage:
# optimizer = HybridDE_SA(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)