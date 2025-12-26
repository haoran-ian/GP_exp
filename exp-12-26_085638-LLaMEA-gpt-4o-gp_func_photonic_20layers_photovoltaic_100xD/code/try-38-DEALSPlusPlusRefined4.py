import numpy as np

class DEALSPlusPlusRefined4:
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
        self.dynamic_F = 0.5
        self.resizing_factor = 0.85
        self.speedup_factor = 1.5

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _levy_flight(self, size, lb, ub):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / beta)
        return np.clip(step, lb, ub)

    def _dual_phase_mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        adaptive_F1 = self.F + np.random.uniform(-0.2, 0.2)
        adaptive_F2 = self.F / (1 + np.exp(-0.05 * (self.evaluations - 0.5 * self.budget)))
        levy_step = self._levy_flight(self.dim, self.lb, self.ub)  # Lévy flight integration
        mutant1 = np.clip(a + adaptive_F1 * (b - c) + levy_step, self.lb, self.ub)  # Enhanced mutation
        mutant2 = np.clip(d + adaptive_F2 * (e - a), self.lb, self.ub)
        return mutant1, mutant2

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

    def _elitist_selection(self):
        elite_size = int(self.elitism_rate * (self.evaluations / self.budget) * self.population_size)
        elite_size = max(elite_size, 2)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def _resize_population(self, iteration):
        if iteration > 0 and iteration % (self.budget // 10) == 0:
            self.population_size = int(self.population_size * self.resizing_factor)
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)

        iteration = 0
        while self.evaluations < self.budget:
            self._resize_population(iteration)
            elite_indices = self._elitist_selection()

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = elite_indices[i % len(elite_indices)]
                target = self.population[target_idx]
                mutant1, mutant2 = self._dual_phase_mutation(target_idx)
                trial1 = self._dynamic_crossover(target, mutant1, iteration)
                trial2 = self._dynamic_crossover(target, mutant2, iteration)

                trial_score1 = func(trial1)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

                trial_score2 = func(trial2)
                self.evaluations += 1

                if trial_score1 < self.scores[target_idx] or trial_score2 < self.scores[target_idx]:
                    better_trial = trial1 if trial_score1 < trial_score2 else trial2
                    better_score = min(trial_score1, trial_score2)
                    self.population[target_idx] = better_trial
                    self.scores[target_idx] = better_score

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