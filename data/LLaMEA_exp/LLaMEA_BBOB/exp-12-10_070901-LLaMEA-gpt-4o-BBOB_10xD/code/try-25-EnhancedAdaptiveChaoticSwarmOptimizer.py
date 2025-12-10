import numpy as np

class EnhancedAdaptiveChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
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

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def non_linear_inertia_weight_decay(self, chaotic_factor):
        return self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * chaotic_factor**2

    def elite_learning_strategy(self, elite_position, position, chaotic_factor):
        return chaotic_factor * (elite_position - position)

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

            diversity = self.calculate_swarm_diversity()
            adaptive_factor = diversity / self.dim
            self.cognitive_coefficient = 1.0 + adaptive_factor
            self.social_coefficient = 2.0 - adaptive_factor
            
            elite_index = np.argmin(self.personal_best_scores)
            elite_position = self.personal_best_positions[elite_index]

            for i in range(self.population_size):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                self.inertia_weight = self.non_linear_inertia_weight_decay(chaotic_factor[i])
                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                elite_learning = self.elite_learning_strategy(elite_position, self.positions[i], chaotic_factor[i])
                
                self.velocities[i] = inertia + cognitive + social + elite_learning
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score