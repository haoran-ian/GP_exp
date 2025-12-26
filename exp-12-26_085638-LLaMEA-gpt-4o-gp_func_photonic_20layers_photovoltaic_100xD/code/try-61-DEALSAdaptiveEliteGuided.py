import numpy as np

class DEALSAdaptiveEliteGuided:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.scores = np.full(self.initial_population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.15
        self.resizing_factor = 0.85

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _elite_guided_mutation(self, idx, elite_indices):
        indices = np.random.choice(elite_indices, 2, replace=False)
        while idx in indices:
            indices = np.random.choice(elite_indices, 2, replace=False)
        a, b = self.population[indices]
        target = self.population[idx]
        mutant = np.clip(target + self.F * (a - b) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        return mutant

    def _adaptive_crossover(self, target, mutant, iteration):
        dynamic_CR = self.CR * (1 - np.cos(iteration / self.budget * np.pi))
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _elitist_selection(self):
        elite_size = int(self.elitism_rate * self.population_size)
        elite_size = max(elite_size, 2)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def _resize_population(self, iteration):
        if iteration > 0 and iteration % (self.budget // 10) == 0:
            self.population_size = int(self.population_size * self.resizing_factor)
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]
        else:
            self.population_size = self.initial_population_size

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)
        self.population_size = self.initial_population_size

        iteration = 0
        while self.evaluations < self.budget:
            self._resize_population(iteration)
            elite_indices = self._elitist_selection()

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = i
                target = self.population[target_idx]
                mutant = self._elite_guided_mutation(target_idx, elite_indices)
                trial = self._adaptive_crossover(target, mutant, iteration)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]