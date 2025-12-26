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
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.chaotic_map_choice = np.random.choice(['logistic', 'sine'])

    def logistic_map(self, x):
        return 4.0 * x * (1 - x)

    def sine_map(self, x):
        return np.sin(np.pi * x)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def chaotic_update(self, x):
        if self.chaotic_map_choice == 'logistic':
            return self.logistic_map(x)
        elif self.chaotic_map_choice == 'sine':
            return self.sine_map(x)

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
            cognitive_coefficient = 1.5 + adaptive_factor
            social_coefficient = 1.5 - adaptive_factor

            for i in range(self.population_size):
                chaotic_factor[i] = self.chaotic_update(chaotic_factor[i])
                inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * chaotic_factor[i]
                inertia = inertia_weight * self.velocities[i]
                cognitive = cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score