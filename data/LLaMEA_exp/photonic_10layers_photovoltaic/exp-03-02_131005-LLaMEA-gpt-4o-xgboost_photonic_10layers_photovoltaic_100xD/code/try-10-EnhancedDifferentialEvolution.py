import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Define search space boundaries
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        # Adaptive parameters
        F_base = 0.5
        CR = 0.9

        while self.evaluations < self.budget:
            F = self._adaptive_F(fitness)
            new_population = []

            for i in range(population_size):
                x = population[i]
                a, b, c = population[np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)]
                y = np.clip(x + F * (a - b + c - x), lb, ub)
                y_mutated = self._mutation(x, y, CR)
                y_fitness = self._evaluate(func, y_mutated)

                if y_fitness < fitness[i]:
                    new_population.append(y_mutated)
                    fitness[i] = y_fitness
                else:
                    new_population.append(x)

            population = np.array(new_population)
            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)

    def _adaptive_F(self, fitness):
        global_best = np.min(fitness)
        global_worst = np.max(fitness)
        diversity = np.std(fitness) / (global_worst - global_best + 1e-6)
        return 0.5 + 0.5 * (1 - diversity)

    def _mutation(self, x, y, CR):
        mutated = np.where(np.random.rand(self.dim) < CR, y, x)
        return mutated