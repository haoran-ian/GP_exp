import numpy as np

class DEALSAdaptiveRefinedEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.1
        self.resizing_factor = 0.85

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _adaptive_mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        adaptive_F = self.F * (1 - np.tanh(self.evaluations / (0.5 * self.budget)))
        elite_avg = np.mean(self.population[self._elitist_selection()], axis=0)
        mutant1 = np.clip(a + adaptive_F * (b - c) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        mutant2 = np.clip(elite_avg + adaptive_F * (e - a) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        return mutant1, mutant2

    def _adaptive_crossover(self, target, mutant, iteration):
        dynamic_CR = self.CR * (1 - np.cos(iteration / self.budget * np.pi))
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _elitist_selection(self):
        dynamic_elitism_rate = self.elitism_rate * (1 + 0.5 * np.sin(np.pi * self.evaluations / self.budget))
        elite_size = int(dynamic_elitism_rate * self.population_size)
        elite_size = max(elite_size, 2)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def _resize_population(self, iteration):
        if iteration > 0 and iteration % (self.budget // 10) == 0:
            self.population_size = int(max(self.population_size * self.resizing_factor, 5))
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
                mutant1, mutant2 = self._adaptive_mutation(target_idx)
                trial1 = self._adaptive_crossover(target, mutant1, iteration)
                trial2 = self._adaptive_crossover(target, mutant2, iteration)

                trial_score1 = func(trial1)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

                trial_score2 = func(trial2)
                self.evaluations += 1

                if min(trial_score1, trial_score2) < self.scores[target_idx]:
                    best_trial = min((trial1, trial_score1), (trial2, trial_score2), key=lambda x: x[1])
                    self.population[target_idx] = best_trial[0]
                    self.scores[target_idx] = best_trial[1]

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]