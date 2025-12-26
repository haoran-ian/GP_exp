import numpy as np

class AdaptiveDEALSPlusRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.15
        self.resizing_factor = 0.85
        self.expansion_factor = 1.2

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _synergy_mutation(self, idx):
        indices = np.random.choice(self.population_size, 4, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 4, replace=False)
        a, b, c, d = self.population[indices]
        F_dynamic = self.F * np.random.uniform(0.5, 1.5)
        mutant = np.clip(a + F_dynamic * (b - c) + F_dynamic * (d - a), self.lb, self.ub)
        return mutant

    def _dynamic_crossover(self, target, mutant, iteration):
        dynamic_CR = np.sin(iteration / self.budget * np.pi) * 0.5 + 0.5
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _elite_focused_local_search(self, elite):
        perturbation_size = 0.02 * (self.ub - self.lb)
        candidate = elite + np.random.uniform(-perturbation_size, perturbation_size, self.dim)
        candidate = np.clip(candidate, self.lb, self.ub)
        return candidate

    def _adaptive_population_resizing(self, iteration):
        resize_interval = self.budget // 10
        if iteration > 0 and iteration % resize_interval == 0:
            improvement_ratio = np.mean(self.scores) / np.min(self.scores)
            if improvement_ratio > 1.0:
                self.population_size = int(self.population_size * self.expansion_factor)
            else:
                self.population_size = int(self.population_size * self.resizing_factor)
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)

        iteration = 0
        while self.evaluations < self.budget:
            self._adaptive_population_resizing(iteration)

            elite_indices = self._elitist_selection()

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = elite_indices[i % len(elite_indices)]
                target = self.population[target_idx]
                mutant = self._synergy_mutation(target_idx)
                trial = self._dynamic_crossover(target, mutant, iteration)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

                if self.evaluations < self.budget:
                    local_candidate = self._elite_focused_local_search(self.population[target_idx])
                    local_score = func(local_candidate)
                    self.evaluations += 1

                    if local_score < self.scores[target_idx]:
                        self.population[target_idx] = local_candidate
                        self.scores[target_idx] = local_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]