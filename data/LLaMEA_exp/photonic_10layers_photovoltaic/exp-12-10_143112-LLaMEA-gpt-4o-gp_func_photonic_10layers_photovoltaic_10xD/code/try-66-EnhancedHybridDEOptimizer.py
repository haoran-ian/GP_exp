import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Initial differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.tau1 = 0.1  # Probability for adjusting F
        self.tau2 = 0.1  # Probability for adjusting CR

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive mutation factor and crossover rate
                if np.random.rand() < self.tau1:
                    self.f = np.random.uniform(0.5, 0.9)
                if np.random.rand() < self.tau2:
                    self.cr = np.random.uniform(0.4, 1.0)

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)

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
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    step_size = (ub - lb) * 0.1 * np.random.rand(self.dim)
                    neighbor = trial + np.random.choice([-1, 1], self.dim) * step_size
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            population[i] = trial
            fitness[i] = trial_fit

        best_idx = np.argmin(fitness)
        return population[best_idx]