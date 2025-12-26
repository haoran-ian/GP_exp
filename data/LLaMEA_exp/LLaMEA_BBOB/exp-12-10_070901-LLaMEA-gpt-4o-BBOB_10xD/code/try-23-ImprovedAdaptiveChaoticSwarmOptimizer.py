import numpy as np

class ImprovedAdaptiveChaoticSwarmOptimizer:
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

    def logistic_map(self, x):
        r = 4.0
        return r * x * (1 - x)

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def rank_based_update(self, i, rank):
        rank_factor = (self.population_size - rank) / self.population_size
        new_position = self.positions[i] + rank_factor * (self.global_best_position - self.positions[i])
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def apply_mutation(self, position):
        mutation_probability = 0.1
        if np.random.rand() < mutation_probability:
            mutation = np.random.uniform(-0.1, 0.1, self.dim)
            position += mutation
        return np.clip(position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        evaluations = 0
        chaotic_factor = np.random.rand(self.population_size)
        
        while evaluations < self.budget:
            scores = []
            for i in range(self.population_size):
                score = func(self.positions[i])
                evaluations += 1
                scores.append((score, i))
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            scores.sort()
            for rank, (score, i) in enumerate(scores):
                chaotic_factor[i] = self.logistic_map(chaotic_factor[i])
                self.inertia_weight = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * chaotic_factor[i]

                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * chaotic_factor[i] * (self.personal_best_positions[i] - self.positions[i])
                social = self.social_coefficient * chaotic_factor[i] * (self.global_best_position - self.positions[i])
                
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] = self.rank_based_update(i, rank)
                self.positions[i] = self.apply_mutation(self.positions[i])

        return self.global_best_position, self.global_best_score