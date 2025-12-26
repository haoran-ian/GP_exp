import numpy as np

class AdaptiveTurbulenceOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.turbulence_factor = 0.05
        self.energy_level = np.ones(self.population_size)
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

        random_turbulence = self.turbulence_factor * np.random.randn(self.population_size, self.dim) * self.energy_level[:, np.newaxis]
        self.velocities = self.inertia_weight * self.velocities + cognitive_velocity + social_velocity + random_turbulence
        self.velocities = np.clip(self.velocities, -self.vel_clamp, self.vel_clamp)

        self.population += self.velocities
        # Update energy levels to maintain diversity dynamically
        self.energy_level = 1 + 0.5 * np.sin(np.pi * np.linspace(0, 1, self.population_size))

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
        # Adapt velocity clamping dynamically
        self.vel_clamp = (0.1 + 0.1 * progress) * (ub - lb)
        # Scale turbulence factor based on progress
        self.turbulence_factor = 0.05 + 0.1 * (1 - progress)

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