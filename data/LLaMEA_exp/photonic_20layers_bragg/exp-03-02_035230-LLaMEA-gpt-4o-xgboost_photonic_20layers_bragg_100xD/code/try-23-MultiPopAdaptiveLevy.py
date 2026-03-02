import numpy as np

class MultiPopAdaptiveLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.9
        self.beta = 0.99
        self.explore_weight = 0.1
        self.num_subpopulations = 3  # Number of subpopulations
        self.subpop_size = self.population_size // self.num_subpopulations

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_budget = self.population_size
        T = 1.0

        while eval_budget < self.budget:
            for subpop_idx in range(self.num_subpopulations):
                subpop_start = subpop_idx * self.subpop_size
                subpop_end = subpop_start + self.subpop_size
                subpop = population[subpop_start:subpop_end]
                subpop_fitness = fitness[subpop_start:subpop_end]

                for i in range(self.subpop_size):
                    a, b, c = subpop[np.random.choice(self.subpop_size, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < self.CR
                    trial = np.where(cross_points, mutant, subpop[i])

                    trial_fitness = func(trial)
                    if eval_budget >= self.budget:
                        break
                    eval_budget += 1
                    if trial_fitness < subpop_fitness[i]:
                        subpop[i] = trial
                        subpop_fitness[i] = trial_fitness
                    else:
                        acceptance_prob = np.exp((subpop_fitness[i] - trial_fitness) / T)
                        if np.random.rand() < acceptance_prob:
                            subpop[i] = trial
                            subpop_fitness[i] = trial_fitness

                subpop_best_idx = np.argmin(subpop_fitness)
                subpop_best = subpop[subpop_best_idx]

                for j in range(self.subpop_size):
                    if np.random.rand() < 0.1:
                        distance = np.linalg.norm(subpop[j] - subpop_best)
                        adjust_factor = np.exp(-self.explore_weight * distance)
                        subpop[j] = subpop[j] + adjust_factor * (subpop_best - subpop[j]) + self.levy_flight(self.dim)
                        subpop[j] = np.clip(subpop[j], bounds[:, 0], bounds[:, 1])
                        subpop_fitness[j] = func(subpop[j])
                        eval_budget += 1
                        if eval_budget >= self.budget:
                            break

            T *= self.alpha * 0.95
            if np.random.rand() < 0.1:
                self.F = self.F * self.beta + self.explore_weight * np.random.rand()
                self.CR = self.CR * (self.beta + 0.01) + self.explore_weight * np.random.rand()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]