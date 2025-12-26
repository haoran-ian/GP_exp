import numpy as np

class EnhancedSwarmOptimizer:
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
        self.neighborhood_size = 5

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)

    def update_particles(self, lb, ub, progress):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        for i in range(self.population_size):
            # Determine the neighborhood for dynamic interaction
            neighbors = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
            local_best = neighbors[np.argmin(self.personal_best_scores[neighbors])]
            neighborhood_best_position = self.personal_best_positions[local_best]

            # Use neighborhood best for social velocity
            cognitive_velocity = self.cognitive_coefficient * r1[i] * (self.personal_best_positions[i] - self.population[i])
            social_velocity = self.social_coefficient * r2[i] * (neighborhood_best_position - self.population[i])

            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
            self.velocities[i] = np.clip(self.velocities[i], -self.vel_clamp, self.vel_clamp)

            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], lb, ub)
        
        # Maintain diversity by reinitializing some particles randomly
        if np.random.rand() < 0.05:
            random_indices = np.random.choice(self.population_size, size=int(0.1 * self.population_size), replace=False)
            self.population[random_indices] = np.random.uniform(lb, ub, (len(random_indices), self.dim))

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
        self.vel_clamp = (0.1 + 0.1 * progress) * (self.global_best_position - self.global_best_position)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)
            self.update_particles(lb, ub, progress)
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.global_best_position, self.global_best_score