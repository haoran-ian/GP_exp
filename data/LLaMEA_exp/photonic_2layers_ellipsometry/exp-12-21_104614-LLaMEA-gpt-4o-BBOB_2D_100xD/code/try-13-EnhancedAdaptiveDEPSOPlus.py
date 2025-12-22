import numpy as np

class EnhancedAdaptiveDEPSOPlus:
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
        self.f = 0.5
        self.cr = 0.9
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.prev_global_best_scores = []
        self.restart_threshold = 1e-3  # Diversity threshold for restart

    def _calculate_diversity(self):
        centroid = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - centroid, axis=1))
        return diversity

    def _adaptive_parameters(self, evaluations):
        if len(self.prev_global_best_scores) > 5:
            recent_improvements = np.diff(self.prev_global_best_scores[-5:])
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement < 1e-6:
                self.f = max(0.4, self.f * 0.95)
                self.w = min(0.8, self.w * 1.05)
            else:
                self.f = min(0.9, self.f * 1.05)
                self.w = max(0.4, self.w * 0.95)

    def _restart_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self._restart_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            diversity = self._calculate_diversity()
            self._adaptive_parameters(evaluations)

            if diversity < self.restart_threshold:
                self._restart_population(lb, ub)

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
                        self.prev_global_best_scores.append(self.global_best_score)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

                if evaluations >= self.budget:
                    break

        return self.global_best