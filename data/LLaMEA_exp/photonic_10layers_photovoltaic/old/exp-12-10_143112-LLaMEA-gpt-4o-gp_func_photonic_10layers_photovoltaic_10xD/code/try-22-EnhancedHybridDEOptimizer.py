import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_f = 0.8
        self.initial_cr = 0.9
        self.local_search_iters = 5

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        f = self.initial_f
        cr = self.initial_cr

        while evaluations < self.budget:
            # Adaptive parameter control
            f = 0.5 + 0.5 * np.random.rand()
            cr = 0.5 + 0.5 * np.random.rand()

            # Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                # Stochastic Nelder-Mead local search
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break
                    neighbor_res = minimize(func, trial + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb),
                                            method='Nelder-Mead', bounds=[(lb[j], ub[j]) for j in range(self.dim)])
                    neighbor = neighbor_res.x
                    neighbor_fit = neighbor_res.fun
                    evaluations += neighbor_res.nfev

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            # Update population with the best found in local search
            population[i] = trial
            fitness[i] = trial_fit

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]