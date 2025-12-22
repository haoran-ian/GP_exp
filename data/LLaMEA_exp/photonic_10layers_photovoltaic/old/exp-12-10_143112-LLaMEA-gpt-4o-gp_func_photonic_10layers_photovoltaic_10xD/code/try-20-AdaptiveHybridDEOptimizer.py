import numpy as np

class AdaptiveHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_min = 0.5  # Minimum mutation factor
        self.f_max = 0.9  # Maximum mutation factor
        self.cr = 0.9  # Crossover probability
        self.dynamic_local_search_iters = 3 * (dim // 10)  # Dynamic based on dimensionality

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential evolution - selection, mutation, and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive mutation factor
                f = self.f_min + (self.f_max - self.f_min) * (1 - evaluations / self.budget)

                mutant = np.clip(a + f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                # Dynamic local search for exploitation
                local_iters = min(self.dynamic_local_search_iters, self.budget - evaluations)
                for _ in range(local_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            # Update population with the best found in local search
            population[i] = trial
            fitness[i] = trial_fit

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]