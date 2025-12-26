import numpy as np
from scipy.optimize import basinhopping
from sklearn.cluster import KMeans
from scipy.stats import levy

class AdaptiveMultiPopDEPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.num_subpopulations = 5
        self.subpopulation_size = self.population_size // self.num_subpopulations
        self.populations = [np.random.uniform(-5, 5, (self.subpopulation_size, dim)) for _ in range(self.num_subpopulations)]
        self.velocities = [np.random.uniform(-1, 1, (self.subpopulation_size, dim)) for _ in range(self.num_subpopulations)]
        self.personal_best = [np.copy(pop) for pop in self.populations]
        self.personal_best_scores = [np.full(self.subpopulation_size, np.inf) for _ in range(self.num_subpopulations)]
        self.global_best = None
        self.global_best_score = np.inf
        self.f = 0.5
        self.cr = 0.9
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.temperature = 1.0

    def _adaptive_parameters(self, evaluations):
        self.temperature = max(0.1, 1.0 - evaluations / self.budget)
        sigmoid_adaptation = 1.0 / (1.0 + np.exp(-0.1 * (evaluations - self.budget / 2)))
        self.f = 0.4 + 0.5 * sigmoid_adaptation
        self.w = 0.4 + 0.5 * sigmoid_adaptation

    def _levy_flight(self, scale=1.0):
        return levy.rvs(size=self.dim) * scale

    def _global_search(self, func, evaluations, lb, ub):
        for pop_idx in range(self.num_subpopulations):
            population = self.populations[pop_idx]
            velocities = self.velocities[pop_idx]
            personal_best = self.personal_best[pop_idx]
            personal_best_scores = self.personal_best_scores[pop_idx]

            for i in range(self.subpopulation_size):
                has_levy_flight = i % 2 == 0
                if has_levy_flight:
                    step = self._levy_flight(scale=self.temperature)
                    trial = np.clip(population[i] + step, lb, ub)
                else:
                    idxs = np.random.choice(self.subpopulation_size, 3, replace=False)
                    x0, x1, x2 = population[idxs]
                    mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, population[i])

                score = func(trial)
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (personal_best[i] - population[i])
                social = self.c2 * r2 * (self.global_best - population[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

        return evaluations

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0

        while evaluations < self.budget:
            self._adaptive_parameters(evaluations)
            evaluations = self._global_search(func, evaluations, lb, ub)

            if evaluations < self.budget and evaluations % 100 == 0:
                for pop_idx in range(self.num_subpopulations):
                    best_idx = np.argmin(self.personal_best_scores[pop_idx])
                    x0 = self.personal_best[pop_idx][best_idx]
                    new_x, new_score = self._local_search(func, x0)
                    evaluations += 5
                    
                    if new_score < self.global_best_score:
                        self.global_best_score = new_score
                        self.global_best = new_x

        return self.global_best

    def _local_search(self, func, x0):
        result = basinhopping(lambda x: func(x), x0, minimizer_kwargs={"method": "Nelder-Mead"}, niter=3)
        return result.x, result.fun