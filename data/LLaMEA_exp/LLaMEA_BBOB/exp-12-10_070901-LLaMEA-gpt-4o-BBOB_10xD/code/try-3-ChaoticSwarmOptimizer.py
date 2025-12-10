import numpy as np

class ChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand(self.population_size)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            for i in range(self.population_size):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                self.inertia_weight = 0.5 + 0.5 * chaotic_factor[i]  # Adapt inertia weight
                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                # Introducing a diversity control mechanism in velocity update
                diversity = np.std(self.positions, axis=0)
                self.velocities[i] = inertia + cognitive + social + 0.1 * diversity
                
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score