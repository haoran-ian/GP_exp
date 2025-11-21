import numpy as np
from scipy.stats import norm

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.current_budget = 0

    def chaotic_map(self, x):
        # Logistic map for generating chaotic sequences
        r = 3.8
        return r * x * (1 - x)

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        # Initialize population with chaotic sequence
        chaotic_sequence = np.zeros(self.population_size)
        chaotic_sequence[0] = 0.7  # Initial value for the logistic map
        for i in range(1, self.population_size):
            chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i-1])
        population = self.lower_bound + chaotic_sequence.reshape(self.population_size, 1) * np.ones(self.dim) * (self.upper_bound - self.lower_bound)
        
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.current_budget < self.budget:
            for i in range(self.population_size):
                if self.current_budget >= self.budget:
                    break

                # Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation factor
                mutant = np.clip(x0 + F * (x1 - x2), self.lower_bound, self.upper_bound)

                # Adaptive Crossover using Gaussian function
                crossover_prob = norm.pdf(self.current_budget/self.budget)
                crossover_mask = np.random.rand(self.dim) < crossover_prob
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial
                trial_fitness = func(trial)
                self.current_budget += 1

                # Simulated Annealing-inspired acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / (1 + self.current_budget / self.budget)):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

        return best

# Example usage:
# optimizer = EnhancedHybridDE_SA(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)