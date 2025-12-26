import numpy as np

class DEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size scales with dimensionality
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population = None
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutate(self, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = self.population[indices]
        mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
        return mutant

    def _crossover(self, target, mutant):
        self.CR = np.random.rand() * 0.5 + 0.5  # Stochastic adaptive CR
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual):
        step_size = 0.1 * (self.ub - self.lb)
        neighbors = individual + np.random.uniform(-step_size, step_size, self.dim)
        neighbors = np.clip(neighbors, self.lb, self.ub)
        return neighbors

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                # Perform adaptive local search
                if self.evaluations < self.budget:
                    local_candidate = self._local_search(self.population[i])
                    local_score = func(local_candidate)
                    self.evaluations += 1

                    if local_score < self.scores[i]:
                        self.population[i] = local_candidate
                        self.scores[i] = local_score

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]