import numpy as np

class EnhancedAdaptivePSO_DE_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.CR = 0.9
        self.F = 0.8
        self.elite_fraction = 0.1
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        # Initialize the population and velocities
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.array([func(ind) for ind in self.population])
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.population[best_idx]
        self.global_best_score = self.personal_best_scores[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Dynamic adaptation of elite fraction
            elite_count = max(1, int((self.elite_fraction + evaluations / (2 * self.budget)) * self.population_size))
            sorted_indices = np.argsort(self.personal_best_scores)
            elites = self.population[sorted_indices[:elite_count]]

            # Dynamic adaptation of inertia weight
            dynamic_inertia = self.inertia_weight * (1 - evaluations / self.budget)

            # PSO update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (
                    dynamic_inertia * self.velocities[i] +
                    self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.population[i]) +
                    self.social_coeff * r2 * (self.global_best_position - self.population[i])
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], lb, ub)

                score = func(self.population[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.population[i]

            # DE mutation and crossover with Multi-scale Adaptive Strategy
            for i in range(self.population_size):
                if i < elite_count:
                    continue

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                trial_vector = np.copy(self.population[i])

                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        # Multi-scale mutation: varying mutation scale based on evaluations
                        scale = 1 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)
                        trial_vector[j] = self.population[a][j] + scale * self.F * (elites[np.random.randint(elite_count)][j] - self.population[c][j])
                        trial_vector[j] = np.clip(trial_vector[j], lb[j], ub[j])

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector

            # Adaptive control of F and CR based on diversity with dynamic scaling
            population_diversity = np.mean(np.std(self.population, axis=0))
            dynamic_scale = (1 + evaluations / self.budget) / 2
            if population_diversity < 0.1:
                self.CR = max(0.5, self.CR * (1 - 0.05 * dynamic_scale))
                self.F = min(0.9, self.F * (1 + 0.05 * dynamic_scale))
            else:
                self.CR = min(0.9, self.CR * (1 + 0.05 * dynamic_scale))
                self.F = max(0.5, self.F * (1 - 0.05 * dynamic_scale))

        return self.global_best_position