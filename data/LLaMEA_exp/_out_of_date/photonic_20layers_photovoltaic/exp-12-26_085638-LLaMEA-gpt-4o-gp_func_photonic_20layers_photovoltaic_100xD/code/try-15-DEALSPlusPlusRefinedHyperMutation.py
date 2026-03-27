import numpy as np

class DEALSPlusPlusRefinedHyperMutation:
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
        self.dynamic_F = 0.5
        self.hypermutation_factor = 0.5  # To adjust hypermutation influence

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def _mutate(self, idx):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = self.population[indices]
        adaptive_F = self.F + np.random.uniform(-0.2, 0.2)
        mutant = np.clip(a + adaptive_F * (b - c), self.lb, self.ub)
        return mutant

    def _hypermutate(self, target, iteration):
        hypermutation_prob = self.hypermutation_factor * (self.budget - self.evaluations) / self.budget
        if np.random.rand() < hypermutation_prob:
            perturbation = np.random.uniform(-0.5, 0.5, self.dim) * (self.ub - self.lb)
            target = np.clip(target + perturbation, self.lb, self.ub)
        return target

    def _dynamic_crossover(self, target, mutant, iteration):
        dynamic_CR = np.sin(iteration / self.budget * np.pi) * 0.5 + 0.5
        cross_points = np.random.rand(self.dim) < dynamic_CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual):
        step_size = 0.05 * (self.ub - self.lb)
        neighbors = individual + np.random.uniform(-step_size, step_size, self.dim)
        neighbors = np.clip(neighbors, self.lb, self.ub)
        return neighbors

    def _elitist_selection(self):
        elite_size = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(self.scores)[:elite_size]
        return elite_indices

    def _crowding_distance(self):
        distances = np.zeros(self.population_size)
        for i in range(self.dim):
            sorted_indices = np.argsort(self.population[:, i])
            sorted_pop = self.population[sorted_indices]
            max_dist = np.max(sorted_pop[:, i]) - np.min(sorted_pop[:, i])
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            for j in range(1, self.population_size - 1):
                distances[sorted_indices[j]] += (sorted_pop[j + 1, i] - sorted_pop[j - 1, i]) / max_dist
        return distances

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        iteration = 0
        while self.evaluations < self.budget:
            crowding_distances = self._crowding_distance()
            elite_indices = np.argsort(crowding_distances)[:int(self.elitism_rate * self.population_size)]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                target_idx = elite_indices[i % len(elite_indices)]
                target = self.population[target_idx]
                mutant = self._mutate(target_idx)
                trial = self._dynamic_crossover(target, mutant, iteration)

                trial = self._hypermutate(trial, iteration)

                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.scores[target_idx]:
                    self.population[target_idx] = trial
                    self.scores[target_idx] = trial_score

                if self.evaluations < self.budget:
                    local_candidate = self._local_search(self.population[target_idx])
                    local_score = func(local_candidate)
                    self.evaluations += 1

                    if local_score < self.scores[target_idx]:
                        self.population[target_idx] = local_candidate
                        self.scores[target_idx] = local_score

            iteration += 1

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]