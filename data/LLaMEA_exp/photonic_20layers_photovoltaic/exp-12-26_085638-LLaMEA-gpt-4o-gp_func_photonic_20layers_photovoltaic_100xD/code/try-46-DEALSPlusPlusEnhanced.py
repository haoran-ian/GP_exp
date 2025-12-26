import numpy as np

class DEALSPlusPlusEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.inertia_weight = 0.7
        self.dynamic_CR = 0.5
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.15
        self.resizing_factor = 0.85

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _adaptive_mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        adaptive_F = self.F * (1 + 0.5 * (1 - self.evaluations / self.budget))
        mutant = np.clip(a + adaptive_F * (b - c) + self.inertia_weight * (d - e), self.lb, self.ub)
        return mutant

    def _adaptive_crossover(self, target, mutant, iteration):
        self.dynamic_CR = 0.5 + 0.5 * np.cos((iteration / self.budget) * np.pi)
        cross_points = np.random.rand(self.dim) < self.dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _resize_population(self, iteration):
        if iteration > 0 and iteration % (self.budget // 10) == 0:
            current_size = len(self.population)
            new_size = max(int(current_size * self.resizing_factor), 5)
            self.population = self.population[:new_size]
            self.scores = self.scores[:new_size]
            self.population_size = new_size

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)

        iteration = 0
        while self.evaluations < self.budget:
            self._resize_population(iteration)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = i
                target = self.population[target_idx]
                mutant = self._adaptive_mutation(target_idx)
                trial = self._adaptive_crossover(target, mutant, iteration)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]