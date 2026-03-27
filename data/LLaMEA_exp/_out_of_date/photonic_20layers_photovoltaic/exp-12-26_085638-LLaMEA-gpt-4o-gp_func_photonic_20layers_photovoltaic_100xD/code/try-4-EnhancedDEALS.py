import numpy as np

class EnhancedDEALS:
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

    def _dynamic_population_adaptation(self):
        # Reduce population size dynamically as evaluations increase
        reduction_factor = 1 - (self.evaluations / self.budget)
        new_population_size = max(5, int(self.population_size * reduction_factor))
        if new_population_size < len(self.population):
            self.population = self.population[:new_population_size]
            self.scores = self.scores[:new_population_size]

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        while self.evaluations < self.budget:
            self._dynamic_population_adaptation()  # Adapt population size
            
            for i in range(len(self.population)):
                if self.evaluations >= self.budget:
                    break

                target = self.population[i]
                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)

                # Multi-trial selection approach
                trial_score = func(trial)
                self.evaluations += 1

                local_candidate = self._local_search(trial)
                if self.evaluations < self.budget:
                    local_score = func(local_candidate)
                    self.evaluations += 1

                best_candidate, best_score = (trial, trial_score) if trial_score < local_score else (local_candidate, local_score)

                if best_score < self.scores[i]:
                    self.population[i] = best_candidate
                    self.scores[i] = best_score

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]