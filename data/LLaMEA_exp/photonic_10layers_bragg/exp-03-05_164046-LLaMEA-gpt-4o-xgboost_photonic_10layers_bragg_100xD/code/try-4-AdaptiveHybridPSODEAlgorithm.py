import numpy as np

class AdaptiveHybridPSODEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.global_best_position = self.particles[0].copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.c1, self.c2 = 1.5, 2.0  # Initial cognitive and social coefficients
        self.w = 0.5  # Initial inertia weight
        self.de_cross_rate = 0.7
        self.de_f = 0.5
        self.evaluations = 0

    def update_velocity(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        self.velocities = np.clip(self.velocities, -0.1, 0.1)

    def update_position(self, bounds):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, bounds.lb, bounds.ub)

    def differential_evolution(self, func, bounds):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            mutant_vector = np.clip(a + self.de_f * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.de_cross_rate
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_value = func(trial_vector)
            self.evaluations += 1
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_position = trial_vector
                    self.global_best_value = trial_value
            if self.evaluations >= self.budget:
                return

    def local_search(self, func, bounds):
        perturbation = np.random.normal(0, 0.01, size=self.dim)
        for i in range(self.population_size):
            candidate = self.particles[i] + perturbation
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            if candidate_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = candidate
                self.personal_best_values[i] = candidate_value
                if candidate_value < self.global_best_value:
                    self.global_best_position = candidate
                    self.global_best_value = candidate_value
            if self.evaluations >= self.budget:
                return

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            self.update_velocity()
            self.update_position(bounds)
            self.differential_evolution(func, bounds)
            self.local_search(func, bounds)
            # Dynamic parameter tuning
            self.w = max(0.2, self.w * 0.99)  # Decrease inertia weight
            self.c1 = max(1.0, self.c1 * 0.99)  # Decrease cognitive coefficient
            self.c2 = min(2.5, self.c2 * 1.01)  # Increase social coefficient
        return self.global_best_position