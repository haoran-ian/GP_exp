import numpy as np

class NeurodynamicChaoticSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.chaotic_factor = np.random.rand(self.population_size)
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.inertia_weight = 0.729
        self.chaotic_intensity = 0.5

    def logistic_map(self, x, r=4.0):
        return r * x * (1 - x)

    def chaotic_perturbation(self, position):
        return position + self.chaotic_intensity * np.random.uniform(-1, 1, size=self.dim)

    def adaptive_neurodynamic_velocity(self, i, diversity):
        c1 = self.c1_initial - diversity * (self.c1_initial - 1.5)
        c2 = self.c2_initial + diversity * (2.5 - self.c2_initial)
        inertia = self.inertia_weight * self.velocities[i]
        cognitive = c1 * np.random.rand() * (self.personal_best_positions[i] - self.positions[i])
        social = c2 * np.random.rand() * (self.global_best_position - self.positions[i])
        return inertia + cognitive + social

    def calculate_swarm_diversity(self):
        return np.mean(np.linalg.norm(self.positions - np.mean(self.positions, axis=0), axis=1))

    def __call__(self, func):
        evaluations = 0
        
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

            for i in range(self.population_size):
                self.chaotic_factor[i] = self.logistic_map(self.chaotic_factor[i])
                self.velocities[i] = self.adaptive_neurodynamic_velocity(i, diversity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = self.chaotic_perturbation(self.positions[i])
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score