import numpy as np

class DEALSEnhanced:
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
        self.feedback_threshold = 0.05  # For adaptive local search
        
    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutate(self, idx, change_group):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = self.population[indices]
        adaptive_F = self.F + np.random.uniform(-0.2, 0.2)  
        mutant = np.clip(a + adaptive_F * (b - c), self.lb, self.ub)
        if change_group:
            dims = np.random.choice(self.dim, size=np.random.randint(1, self.dim), replace=False)
            mutant[dims] = a[dims] + adaptive_F * (b[dims] - c[dims])
        return mutant

    def _crossover(self, target, mutant):
        self.CR = np.random.rand() * 0.5 + 0.5
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual, score):
        step_size = 0.05 * (self.ub - self.lb)
        neighbors = individual + np.random.uniform(-step_size, step_size, self.dim)
        neighbors = np.clip(neighbors, self.lb, self.ub)
        new_score = func(neighbors)
        if new_score < score - self.feedback_threshold:
            return neighbors, new_score
        return individual, score

    def _elitist_selection(self):
        elite_size = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def _resize_population(self, iteration):
        if iteration > 0 and iteration % (self.budget // 10) == 0:
            reduction_factor = 0.85
            self.population_size = int(self.population_size * reduction_factor)
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        iteration = 0
        while self.evaluations < self.budget:
            self._resize_population(iteration)
            elite_indices = self._elitist_selection()

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = elite_indices[i % len(elite_indices)]
                target = self.population[target_idx]
                mutate_group = (i % 2 == 0)
                mutant = self._mutate(target_idx, mutate_group)
                trial = self._crossover(target, mutant)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

                if self.evaluations < self.budget:
                    local_candidate, local_score = self._local_search(self.population[target_idx], self.scores[target_idx])
                    self.evaluations += 1

                    if local_score < self.scores[target_idx]:
                        self.population[target_idx] = local_candidate
                        self.scores[target_idx] = local_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]