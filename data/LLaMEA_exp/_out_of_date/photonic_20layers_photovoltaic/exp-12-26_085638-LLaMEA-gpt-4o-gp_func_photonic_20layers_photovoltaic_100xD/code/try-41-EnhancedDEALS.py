import numpy as np

class EnhancedDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.8
        self.CR = 0.9
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.elitism_rate = 0.15
        self.resizing_factor = 0.85
        self.learning_rate = 0.1
        self.adaptive_F = [0.5, 0.7, 0.9]  # Different mutation strategies

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.lb, self.ub = lb, ub

    def _multi_rate_mutation(self, idx):
        indices = np.random.choice(self.population_size, 5, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 5, replace=False)
        a, b, c, d, e = self.population[indices]
        mutants = []
        for F in self.adaptive_F:
            mutants.append(np.clip(a + F * (b - c) + np.random.normal(0, 0.05, self.dim), self.lb, self.ub))
            mutants.append(np.clip(d + F * (e - a), self.lb, self.ub))
        return mutants

    def _dynamic_crossover(self, target, mutant, iteration):
        dynamic_CR = np.cos(iteration / self.budget * np.pi) * 0.5 + 0.5
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _elitist_selection(self):
        elite_size = int(self.elitism_rate * self.population_size)
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
                mutants = self._multi_rate_mutation(target_idx)
                trials = [self._dynamic_crossover(target, mutant, iteration) for mutant in mutants]

                trial_scores = [func(trial) for trial in trials]
                self.evaluations += len(trials)

                best_trial_index = np.argmin(trial_scores)
                best_trial = trials[best_trial_index]
                best_score = trial_scores[best_trial_index]

                if best_score < self.scores[target_idx]:
                    self.population[target_idx] = best_trial
                    self.scores[target_idx] = best_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]