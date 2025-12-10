import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_min, self.f_max = 0.5, 0.9  # Adaptive mutation factor range
        self.cr_min, self.cr_max = 0.1, 0.9  # Adaptive crossover probability range
        self.local_search_iters = 5
        self.evaluations = 0

    def adaptive_parameters(self):
        t = self.evaluations / self.budget
        f = self.f_min + (self.f_max - self.f_min) * (1 - t)
        cr = self.cr_min + (self.cr_max - self.cr_min) * t
        return f, cr

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                f, cr = self.adaptive_parameters()
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                self.evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                for _ in range(self.local_search_iters):
                    if self.evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.normal(0, 0.1, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    self.evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

                population[i] = trial
                fitness[i] = trial_fit

        best_idx = np.argmin(fitness)
        return population[best_idx]