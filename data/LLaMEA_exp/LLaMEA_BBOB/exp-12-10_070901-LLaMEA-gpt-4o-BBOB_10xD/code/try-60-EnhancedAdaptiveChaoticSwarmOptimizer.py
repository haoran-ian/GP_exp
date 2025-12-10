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
        self.cognitive_coefficient_min = 1.0
        self.cognitive_coefficient_max = 2.0
        self.social_coefficient_min = 1.0
        self.social_coefficient_max = 2.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def levy_flight(self, size, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v) ** (1 / beta)

    def chaotic_mutation(self, position, factor):
        mutation_scale = 0.05
        chaotic_value = self.logistic_map(factor)
        return position + mutation_scale * chaotic_value * np.random.uniform(-1, 1, size=self.dim)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def adaptive_velocity_scaling(self, diversity):
        return (self.inertia_weight_min + diversity) / (1 + diversity)

    def update_learning_coefficients(self, diversity):
        adaptive_factor = diversity / self.dim
        self.cognitive_coefficient = self.cognitive_coefficient_min + adaptive_factor * (self.cognitive_coefficient_max - self.cognitive_coefficient_min)
        self.social_coefficient = self.social_coefficient_max - adaptive_factor * (self.social_coefficient_max - self.social_coefficient_min)

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
            self.update_learning_coefficients(diversity)

            for i in range(self.population_size):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                
                # Updated line with adaptive inertia weight strategy
                self.inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * chaotic_factor[i] * (1 - diversity / self.dim) 
                
                inertia = self.adaptive_velocity_scaling(diversity) * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                levy_prob = 0.1 + 0.4 * (1 - diversity / self.dim)
                if np.random.rand() < levy_prob:
                    levy_step = 0.01 * self.levy_flight(self.dim)
                    self.positions[i] += levy_step

                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = self.chaotic_mutation(self.positions[i], chaotic_factor[i])
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score