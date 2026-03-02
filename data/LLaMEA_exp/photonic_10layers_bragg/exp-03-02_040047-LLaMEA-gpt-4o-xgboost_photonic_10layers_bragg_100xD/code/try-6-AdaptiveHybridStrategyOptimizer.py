import numpy as np

class AdaptiveHybridStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.explore_exploit_ratio = 0.5
        self.adaptive_factor = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = population_size
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            
            for i in range(population_size):
                if np.random.rand() < self.explore_exploit_ratio:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + self.scaling_factor * (b - c), lb, ub)
                else:
                    if np.random.rand() < 0.5:
                        mutant = best_solution + np.random.normal(0, self.adaptive_factor, self.dim)
                    else:
                        perturb = np.random.uniform(-self.adaptive_factor, self.adaptive_factor, self.dim)
                        mutant = best_solution + perturb
                    mutant = np.clip(mutant, lb, ub)

                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])
                new_population[i] = np.clip(trial, lb, ub)

            new_fitness = np.apply_along_axis(func, 1, new_population)
            evaluations += population_size

            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_solution = population[np.argmin(fitness)]
                self.adaptive_factor = max(0.1, self.adaptive_factor * 0.9)

            if evaluations >= self.budget / 2 and evaluations % (self.initial_population_size * 10) == 0:
                population_size = min(population_size * 2, int(self.budget / 10))
                population = np.vstack((population, np.random.uniform(lb, ub, (population_size - len(population), self.dim))))
                fitness = np.hstack((fitness, np.apply_along_axis(func, 1, population[len(fitness):])))
                evaluations += population_size - len(fitness)

        return best_solution