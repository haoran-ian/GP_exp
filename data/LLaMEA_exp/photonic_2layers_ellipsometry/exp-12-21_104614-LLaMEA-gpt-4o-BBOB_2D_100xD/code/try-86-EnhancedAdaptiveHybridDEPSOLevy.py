import numpy as np
from scipy.optimize import basinhopping
from scipy.stats import levy

class EnhancedAdaptiveHybridDEPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = np.inf
        self.f_base = 0.5
        self.cr = 0.9
        self.w_base = 0.9
        self.c1 = 2.0
        self.c2 = 2.0

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adaptive_parameters(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.w = self.w_base - 0.5 * progress_ratio
        self.f = self.f_base + 0.3 * np.sin(2 * np.pi * progress_ratio)

    def _levy_flight(self, scale=1.0):
        return levy.rvs(size=self.dim) * scale

    def _local_search(self, func, x0):
        result = basinhopping(lambda x: func(x), x0, minimizer_kwargs={"method": "Nelder-Mead"}, niter=5)
        return result.x, result.fun

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            self._adaptive_parameters(evaluations)

            diversity = self._calculate_diversity()
            levy_scale = np.clip((diversity / self.dim), 0.1, 1.0)

            for i in range(self.population_size):
                has_levy_flight = i % 3 == 0
                if has_levy_flight:
                    step = self._levy_flight(scale=levy_scale)
                    trial = np.clip(self.population[i] + step, lb, ub)
                else:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = self.population[idxs]
                    mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cr
                    trial = np.where(crossover, mutant, self.population[i])

                if evaluations < self.budget:
                    score = func(trial)
                    evaluations += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best[i] = trial
                        if score < self.global_best_score:
                            self.global_best_score = score
                            self.global_best = trial

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget and evaluations % 100 == 0:
                best_idx = np.argmin(self.personal_best_scores)
                x0 = self.personal_best[best_idx]
                new_x, new_score = self._local_search(func, x0)
                evaluations += 5
                if new_score < self.global_best_score:
                    self.global_best_score = new_score
                    self.global_best = new_x

        elite_population_idx = np.argsort(self.personal_best_scores)[:5]
        for idx in elite_population_idx:
            self.population[idx] = np.clip(self.global_best + 0.1 * np.random.randn(self.dim), lb, ub)

        return self.global_best