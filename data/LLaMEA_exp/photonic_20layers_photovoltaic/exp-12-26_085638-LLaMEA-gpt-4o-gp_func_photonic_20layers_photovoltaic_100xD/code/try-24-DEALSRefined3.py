import numpy as np

class DEALSRefined3:
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
        self.diversity_factor = 0.6

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _diversity_oriented_mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        F1 = np.random.uniform(self.F - 0.2, self.F + 0.2)
        F2 = self.F * (1.0 + np.sin(2 * np.pi * (self.evaluations / self.budget)))
        mutant1 = np.clip(a + F1 * (b - c + self.diversity_factor * (d - e)), self.lb, self.ub)
        mutant2 = np.clip(d + F2 * (e - a + self.diversity_factor * (b - c)), self.lb, self.ub)
        return mutant1, mutant2

    def _adaptive_crossover(self, target, mutant, iteration):
        adaptive_CR = (0.5 + 0.5 * np.sin(iteration / self.budget * np.pi))
        cross_points = np.random.rand(self.dim) < adaptive_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, candidate):
        perturbation_size = 0.01 * (self.ub - self.lb)
        neighbor = candidate + np.random.uniform(-perturbation_size, perturbation_size, self.dim)
        neighbor = np.clip(neighbor, self.lb, self.ub)
        return neighbor

    def _selection_pressure(self, iteration):
        return max(0.1, 1.0 - iteration / self.budget)

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)

        iteration = 0
        while self.evaluations < self.budget:
            elite_size = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.scores)[:elite_size]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = elite_indices[i % len(elite_indices)]
                target = self.population[target_idx]
                mutant1, mutant2 = self._diversity_oriented_mutation(target_idx)
                trial1 = self._adaptive_crossover(target, mutant1, iteration)
                trial2 = self._adaptive_crossover(target, mutant2, iteration)

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
                    local_candidate = self._local_search(self.population[target_idx])
                    local_score = func(local_candidate)
                    self.evaluations += 1

                    if local_score < self.scores[target_idx]:
                        self.population[target_idx] = local_candidate
                        self.scores[target_idx] = local_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]