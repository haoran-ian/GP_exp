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
    
    def tent_map(self, x):
        return 1 - np.abs(1 - 2 * x)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def __call__(self, func):
        evaluations = 0
        chaotic_factors = np.random.rand(self.population_size, 2)
        
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

            for i in range(self.population_size):
                chaotic_factors[i, 0] = self.logistic_map(chaotic_factors[i, 0])
                chaotic_factors[i, 1] = self.tent_map(chaotic_factors[i, 1])
                chaotic_mix = (chaotic_factors[i, 0] + chaotic_factors[i, 1]) / 2
                self.inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * chaotic_mix
                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_mix * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_mix * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = inertia + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)  # Velocity clamping
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score