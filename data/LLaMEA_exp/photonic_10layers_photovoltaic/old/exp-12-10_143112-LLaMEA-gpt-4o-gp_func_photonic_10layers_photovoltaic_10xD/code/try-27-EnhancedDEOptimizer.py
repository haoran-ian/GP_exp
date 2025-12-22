import numpy as np

class EnhancedDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_min, self.f_max = 0.5, 1.0  # Adaptive differential mutation factor range
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential evolution - adaptive mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.f_min + np.random.rand() * (self.f_max - self.f_min)
                mutant = np.clip(a + F_dynamic * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                # Strategic local search for exploitation
                if np.random.rand() < 0.5:  # Probability to perform local search
                    for _ in range(self.local_search_iters):
                        if evaluations >= self.budget:
                            break

                        neighbor_step = np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
                        neighbor = trial + neighbor_step
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