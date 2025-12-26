import numpy as np

class AdaptivePrioritySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.diversification_factor = 0.1

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)

    def update_particles(self, lb, ub):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        for i in range(self.population_size):
            cognitive_velocity = self.cognitive_coefficient * r1[i] * (self.personal_best_positions[i] - self.population[i])
            social_velocity = self.social_coefficient * r2[i] * (self.global_best_position - self.population[i])

            # Adaptive priority based on personal best scores
            priority = (self.personal_best_scores.max() - self.personal_best_scores[i]) / (self.personal_best_scores.max() - self.personal_best_scores.min() + 1e-9)
            inertia_weighted_velocity = self.inertia_weight * self.velocities[i]
            self.velocities[i] = inertia_weighted_velocity + priority * (cognitive_velocity + social_velocity)

            # Diversification influence based on convergence trends
            if np.random.rand() < self.diversification_factor:
                random_displacement = np.random.uniform(-1, 1, self.dim)
                self.velocities[i] += 0.2 * np.abs(ub - lb) * random_displacement

            self.velocities[i] = np.clip(self.velocities[i], -self.vel_clamp, self.vel_clamp)
            self.population[i] = np.clip(self.population[i] + self.velocities[i], lb, ub)

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
        self.inertia_weight = self.inertia_min + (0.9 - self.inertia_min) * (1 - progress)
        self.cognitive_coefficient = 2.0 - 1.5 * progress
        self.social_coefficient = 1.0 + 1.5 * progress
        self.diversification_factor = 0.05 + 0.45 * progress

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)
            self.update_particles(lb, ub)
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.global_best_position, self.global_best_score