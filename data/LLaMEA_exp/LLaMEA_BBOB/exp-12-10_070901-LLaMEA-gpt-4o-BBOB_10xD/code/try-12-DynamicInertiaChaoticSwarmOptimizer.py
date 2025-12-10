import numpy as np

class DynamicInertiaChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.max_population_size = 40
        self.min_population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def mutate_positions(self):
        mutation_strength = 0.1 * (self.upper_bound - self.lower_bound) * (1 - self.evaluations / self.budget)
        mutation = np.random.uniform(-mutation_strength, mutation_strength, self.positions.shape)
        self.positions += mutation

    def update_population_size(self):
        diversity = self.calculate_swarm_diversity()
        if diversity < 0.1:
            new_size = max(self.min_population_size, self.positions.shape[0] - 1)
        else:
            new_size = min(self.max_population_size, self.positions.shape[0] + 1)
        if new_size != self.positions.shape[0]:
            indices = np.argsort(self.personal_best_scores)[:new_size]
            self.positions = self.positions[indices]
            self.velocities = self.velocities[indices]
            self.personal_best_positions = self.personal_best_positions[indices]
            self.personal_best_scores = self.personal_best_scores[indices]

    def __call__(self, func):
        self.evaluations = 0
        chaotic_factor = np.random.rand(self.positions.shape[0])
        
        while self.evaluations < self.budget:
            for i in range(self.positions.shape[0]):
                score = func(self.positions[i])
                self.evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            self.update_population_size()
            
            for i in range(self.positions.shape[0]):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                inertia_weight = self.initial_inertia_weight * (1 - (self.evaluations / self.budget))
                inertia = inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            self.mutate_positions()

        return self.global_best_position, self.global_best_score