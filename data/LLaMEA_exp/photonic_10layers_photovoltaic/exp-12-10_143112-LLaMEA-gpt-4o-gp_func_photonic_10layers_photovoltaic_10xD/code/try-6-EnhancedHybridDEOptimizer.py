import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adapt F and CR dynamically
                f = np.random.uniform(0.5, 1.0)
                cr = np.random.uniform(0.7, 1.0)

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

                # Enhanced Local search using neighborhood exploration
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbors = [trial + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb) for _ in range(3)]
                    neighbors = [np.clip(neighbor, lb, ub) for neighbor in neighbors]
                    neighbors_fit = [func(neighbor) for neighbor in neighbors]
                    evaluations += len(neighbors)

                    best_neighbor_idx = np.argmin(neighbors_fit)
                    if neighbors_fit[best_neighbor_idx] < trial_fit:
                        trial = neighbors[best_neighbor_idx]
                        trial_fit = neighbors_fit[best_neighbor_idx]

            # Update population with the best found in local search
            population[i] = trial
            fitness[i] = trial_fit

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]