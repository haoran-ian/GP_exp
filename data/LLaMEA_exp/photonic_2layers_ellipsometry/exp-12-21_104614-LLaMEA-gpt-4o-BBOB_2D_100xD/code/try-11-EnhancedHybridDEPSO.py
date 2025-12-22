import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = np.inf
        self.f = 0.5  # Scaling factor for differential evolution
        self.cr = 0.9  # Crossover probability for differential evolution
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 2.0  # Cognitive (personal) component for PSO
        self.c2 = 2.0  # Social (global) component for PSO

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adjust_population_size(self, evaluations, max_evaluations):
        convergence_rate = (self.global_best_score - np.min(self.personal_best_scores)) / self.global_best_score
        diversity = self._calculate_diversity()
        if convergence_rate < 0.1 and diversity < 0.1:
            self.population_size = min(int(self.population_size * 1.1), 2 * self.initial_population_size)
        elif evaluations > 0.5 * max_evaluations and diversity > 0.2:
            self.population_size = max(int(self.population_size * 0.9), self.initial_population_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            self._adjust_population_size(evaluations, self.budget)
            diversity = self._calculate_diversity()
            self.w = 0.1 + 0.4 * (1 - diversity / (np.linalg.norm(ub - lb) / 2))
            self.f = 0.6 + 0.2 * (1 - diversity / (np.linalg.norm(ub - lb) / 2))

            for i in range(self.population_size):
                while True:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    if i not in idxs:
                        break
                x0, x1, x2 = self.population[idxs]
                mutant = np.clip(x0 + self.f * (x1 - x2), lb, ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])

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

        return self.global_best