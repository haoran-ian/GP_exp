import numpy as np

class EnhancedChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.adaptive_chaotic_factor = np.random.rand(self.initial_population_size)

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def __call__(self, func):
        evaluations = 0
        population_size = self.initial_population_size

        while evaluations < self.budget:
            # Dynamic adjustment of population size
            if evaluations % (self.budget // 10) == 0 and population_size < self.budget // 2:
                population_size += 1
                self.positions = np.vstack((self.positions, np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))))
                self.velocities = np.vstack((self.velocities, np.random.uniform(-1, 1, (1, self.dim))))
                self.personal_best_positions = np.vstack((self.personal_best_positions, self.positions[-1]))
                self.personal_best_scores = np.append(self.personal_best_scores, np.inf)
                self.adaptive_chaotic_factor = np.append(self.adaptive_chaotic_factor, np.random.rand())

            for i in range(population_size):
                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            for i in range(population_size):
                self.adaptive_chaotic_factor[i] = self.logistic_map(self.adaptive_chaotic_factor[i])
                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * self.adaptive_chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * self.adaptive_chaotic_factor[i] * (self.global_best_position - self.positions[i])

                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score