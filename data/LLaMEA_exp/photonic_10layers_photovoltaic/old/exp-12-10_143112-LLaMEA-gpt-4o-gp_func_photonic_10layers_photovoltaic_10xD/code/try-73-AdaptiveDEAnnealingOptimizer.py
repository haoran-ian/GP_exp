import numpy as np

class AdaptiveDEAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.initial_temperature = 1.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        temperature = self.initial_temperature

        while evaluations < self.budget:
            # Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                adaptive_f = self.f * (1 - (evaluations / self.budget))
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                # Simulated annealing acceptance criterion
                delta = trial_fit - fitness[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                    population[i] = trial
                    fitness[i] = trial_fit

                temperature *= self.cooling_rate

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]