import numpy as np

class EnhancedDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Initial population size
        self.min_population_size = 5 * dim  # Minimum population size allowed
        self.max_population_size = 15 * dim  # Maximum population size allowed
        self.F = 0.8  # Initial differential weight
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

    def _dynamic_population_size(self):
        # Dynamically adjust population size based on convergence
        if self.scores.std() < 0.01:  # If convergence is detected
            self.population_size = max(self.min_population_size, self.population_size - self.dim)
        else:
            self.population_size = min(self.max_population_size, self.population_size + self.dim)

    def _adjust_F(self):
        # Dynamically adjust F based on performance
        self.F = np.clip(0.5 + 0.5 * (1 - np.mean(self.scores) / (np.std(self.scores) + 1e-8)), 0.4, 0.9)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        while self.evaluations < self.budget:
            self._dynamic_population_size()
            self._adjust_F()

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