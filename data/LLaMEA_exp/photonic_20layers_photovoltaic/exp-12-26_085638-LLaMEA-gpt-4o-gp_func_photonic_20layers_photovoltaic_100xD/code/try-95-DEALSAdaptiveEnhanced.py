import numpy as np

class DEALSAdaptiveEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_F = 0.8
        self.initial_CR = 0.9
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.15
        self.resizing_factor = 0.85
        self.learning_rate = 0.1

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _dynamic_parameters(self, iteration):
        adaptive_F = self.initial_F * (1 - np.tanh(self.evaluations / (0.5 * self.budget)))
        dynamic_CR = self.initial_CR * (1 - np.cos(iteration / self.budget * np.pi))
        return adaptive_F, dynamic_CR

    def _adaptive_mutation(self, idx, adaptive_F):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        elite_avg = np.mean(self.population[self._elitist_selection()], axis=0)
        mutant = np.clip(a + adaptive_F * (b - c) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        return mutant

    def _adaptive_crossover(self, target, mutant, dynamic_CR):
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _rank_based_selection(self):
        sorted_indices = np.argsort(self.scores)
        rank_probabilities = np.linspace(1, 0, self.population_size)
        rank_probabilities /= rank_probabilities.sum()
        return np.random.choice(sorted_indices, p=rank_probabilities)

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
            adaptive_F, dynamic_CR = self._dynamic_parameters(iteration)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = self._rank_based_selection()
                target = self.population[target_idx]
                mutant = self._adaptive_mutation(target_idx, adaptive_F)
                trial = self._adaptive_crossover(target, mutant, dynamic_CR)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]