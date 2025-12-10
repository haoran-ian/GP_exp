import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.f_min, self.f_max = 0.5, 0.9  # Adaptive mutation factor range
        self.cr_min, self.cr_max = 0.1, 0.9  # Adaptive crossover probability range
        self.local_search_iters = 5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = evaluations / self.budget
            self.f = self.f_min + adaptive_factor * (self.f_max - self.f_min)
            self.cr = self.cr_max - adaptive_factor * (self.cr_max - self.cr_min)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

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

                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            if evaluations < self.budget:
                population[i] = trial
                fitness[i] = trial_fit

            if evaluations >= 0.7 * self.budget:
                # Reduce population size towards the end to focus on exploitation
                self.population_size = max(5, int(self.initial_population_size * (1 - adaptive_factor)))

        best_idx = np.argmin(fitness)
        return population[best_idx]