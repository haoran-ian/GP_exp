import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.CR = 0.9
        self.F = 0.8
        self.inertia_weight = 0.5
        self.cognitive = 1.5
        self.social = 1.5
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_CR = self.CR
        self.adaptive_F = self.F
        self.adaptive_inertia_weight = self.inertia_weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            self.adaptive_CR = 0.9 - 0.5 * (evaluations / self.budget)
            self.adaptive_F = 0.8 + 0.4 * (evaluations / self.budget)
            self.adaptive_inertia_weight = 0.5 + 0.3 * (1 - evaluations / self.budget)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = np.arange(self.population_size)
                np.random.shuffle(indices)
                indices = indices[indices != i]
                a, b, c = self.population[indices[:3]]
                mutant_vector = a + self.adaptive_F * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                crossover_mask = np.random.rand(self.dim) < self.adaptive_CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.best_scores[i]:
                    self.best_scores[i] = trial_score
                    self.best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive * r1 * (self.best_positions - self.population)
            social_component = self.social * r2 * (self.global_best_position - self.population)
            self.velocities = self.adaptive_inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, lb, ub)

        return self.global_best_position, self.global_best_score