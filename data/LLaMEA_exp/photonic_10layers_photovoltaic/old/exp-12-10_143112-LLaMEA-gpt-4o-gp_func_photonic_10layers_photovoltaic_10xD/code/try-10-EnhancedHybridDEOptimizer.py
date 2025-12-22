import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f_min = 0.5  # Minimum mutation factor
        self.f_max = 1.0  # Maximum mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.adaptive_rate = 0.05  # Rate for adaptive strategy

    def adaptive_mutation_factor(self, fitness, idx):
        rank = np.argsort(fitness)
        scale = (self.f_max - self.f_min) * (rank[idx] / (self.population_size - 1))
        return self.f_max - scale

    def guided_local_search(self, current_best, trial, lb, ub):
        direction = current_best - trial
        step = np.random.uniform(0.01, 0.1, self.dim) * direction
        guided_neighbor = np.clip(trial + step, lb, ub)
        return guided_neighbor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                f_i = self.adaptive_mutation_factor(fitness, i)
                mutant = np.clip(a + f_i * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                current_best = population[np.argmin(fitness)]
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    neighbor = self.guided_local_search(current_best, trial, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit:
                        trial = neighbor
                        trial_fit = neighbor_fit

            population[i] = trial
            fitness[i] = trial_fit

        best_idx = np.argmin(fitness)
        return population[best_idx]