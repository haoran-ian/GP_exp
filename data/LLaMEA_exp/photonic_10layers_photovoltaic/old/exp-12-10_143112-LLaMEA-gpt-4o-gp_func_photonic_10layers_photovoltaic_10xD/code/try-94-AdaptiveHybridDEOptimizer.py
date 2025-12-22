import numpy as np

class AdaptiveHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_min, self.f_max = 0.5, 0.9  # Adaptive differential mutation factor range
        self.cr_min, self.cr_max = 0.3, 0.9  # Adaptive crossover probability range
        self.local_search_iters = 5

    def adaptive_params(self, evaluations):
        # Adapt mutation factor and crossover probability based on the progress
        progress = evaluations / self.budget
        f = self.f_min + progress * (self.f_max - self.f_min)
        cr = self.cr_max - progress * (self.cr_max - self.cr_min)
        return f, cr

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            f, cr = self.adaptive_params(evaluations)
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

                dynamic_step_size = 0.1 * (1 - evaluations / self.budget)
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-dynamic_step_size, dynamic_step_size, self.dim) * (ub - lb)
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