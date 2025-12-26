import numpy as np

class RefinedDynamicNeighborhoodSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.vel_clamp = None
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)

    def update_particles(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        cognitive_velocity = self.cognitive_coefficient * r1 * (self.personal_best_positions - self.population)
        social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.population)

        sorted_indices = np.argsort(self.personal_best_scores)
        elite_indices = sorted_indices[:5]  # Top 5 elites
        for i in range(self.population_size):
            neighbors = sorted_indices[max(0, i - 2):min(self.population_size, i + 3)]
            local_leader_velocity = r1[i] * (np.mean(self.personal_best_positions[neighbors], axis=0) - self.population[i])
            elite_guided_velocity = r2[i] * (np.mean(self.personal_best_positions[elite_indices], axis=0) - self.population[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_velocity[i] + social_velocity[i] +
                                  local_leader_velocity + elite_guided_velocity)
            self.velocities[i] = np.clip(self.velocities[i], -self.vel_clamp, self.vel_clamp)

        self.population += self.velocities

    def evaluate_population(self, func):
        for i in range(self.population_size):
            score = func(self.population[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.population[i]

    def adapt_parameters(self, progress):
        if progress < 0.5:
            self.inertia_weight = self.inertia_min + (0.9 - self.inertia_min) * (1 - progress * 2)
            self.cognitive_coefficient = 2.5 - 2.0 * progress
            self.social_coefficient = 0.5 + 2.0 * progress
        else:
            self.inertia_weight = self.inertia_min
            self.cognitive_coefficient = 1.0 + progress
            self.social_coefficient = 2.5 - 1.5 * progress
        self.vel_clamp = (0.1 + 0.2 * progress) * (ub - lb)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)
            self.update_particles()
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.global_best_position, self.global_best_score