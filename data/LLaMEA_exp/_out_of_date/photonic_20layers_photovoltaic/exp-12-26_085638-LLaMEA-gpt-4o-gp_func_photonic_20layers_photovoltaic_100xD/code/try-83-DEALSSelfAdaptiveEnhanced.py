import numpy as np

class DEALSSelfAdaptiveEnhanced:
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
        self.alpha_min = 0.1
        self.alpha_max = 0.9
        self.beta = 0.1

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _self_adaptive_parameters(self):
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (1 - self.evaluations / self.budget)
        return alpha

    def _elitist_avg(self):
        elite_indices = self._elitist_selection()
        return np.mean(self.population[elite_indices], axis=0)

    def _mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        alpha = self._self_adaptive_parameters()
        elite_avg = self._elitist_avg()
        F_dynamic = self.F * (1 - np.tanh(self.evaluations / (0.5 * self.budget)))
        mutant1 = np.clip(a + F_dynamic * (b - c) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        mutant2 = np.clip(elite_avg + F_dynamic * (e - a) * alpha + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
        return mutant1, mutant2

    def _crossover(self, target, mutant, iteration):
        dynamic_CR = self.CR * (1 - np.cos(iteration / self.budget * np.pi))
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _elitist_selection(self):
        elite_size = max(int(self.elitism_rate * (self.evaluations / self.budget) * self.population_size), 2)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def __call__(self, func):
        self._initialize_population(func.bounds.lb, func.bounds.ub)

        iteration = 0
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant1, mutant2 = self._mutation(i)
                trial1 = self._crossover(target, mutant1, iteration)
                trial2 = self._crossover(target, mutant2, iteration)

                trial_score1 = func(trial1)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

                trial_score2 = func(trial2)
                self.evaluations += 1

                if min(trial_score1, trial_score2) < self.scores[i]:
                    best_trial = min((trial1, trial_score1), (trial2, trial_score2), key=lambda x: x[1])
                    self.population[i] = best_trial[0]
                    self.scores[i] = best_trial[1]

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]