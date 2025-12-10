import numpy as np

class EnhancedAdaptiveChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        return 4.0 * x * (1 - x)

    def tent_map(self, x):
        return 2 * x if x < 0.5 else 2 * (1 - x)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand(self.initial_population_size)
        
        while evaluations < self.budget:
            for i in range(self.initial_population_size):
                score = func(self.positions[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            diversity = self.calculate_swarm_diversity()
            population_size = max(4, int(self.initial_population_size * (1 - diversity)))
            self.positions = self.positions[:population_size]
            self.velocities = self.velocities[:population_size]
            self.personal_best_positions = self.personal_best_positions[:population_size]
            self.personal_best_scores = self.personal_best_scores[:population_size]
            chaotic_factor = chaotic_factor[:population_size]

            for i in range(population_size):
                if evaluations % 2 == 0:
                    chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                else:
                    chaotic_factor[i] = self.tent_map(chaotic_factor[i])
                    
                inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * chaotic_factor[i]
                inertia = inertia_weight * self.velocities[i]
                cognitive = 1.5 * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = 1.5 * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score