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
        self.c1, self.c2 = 1.5, 2.0  # cognitive and social coefficients
        self.w = 0.5  # inertia weight
        self.de_cross_rate = 0.7
        self.de_f = 0.5
        self.evaluation_progress = 0

    def adapt_parameters(self):
        progress_ratio = self.evaluation_progress / self.budget
        self.w = 0.9 - progress_ratio * 0.5
        self.de_f = 0.5 + progress_ratio * 0.4
        self.de_cross_rate = 0.7 - progress_ratio * 0.2

    def update_velocity(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive_component + social_component

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
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_position = trial_vector
                    self.global_best_value = trial_value

    def __call__(self, func):
        evaluations = 0
        bounds = func.bounds
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                evaluations += 1
                self.evaluation_progress = evaluations
                if fitness < self.personal_best_values[i]:
                    self.personal_best_positions[i] = self.particles[i].copy()
                    self.personal_best_values[i] = fitness
                    if fitness < self.global_best_value:
                        self.global_best_position = self.particles[i].copy()
                        self.global_best_value = fitness
                if evaluations >= self.budget:
                    return self.global_best_position
            self.adapt_parameters()
            self.update_velocity()
            self.update_position(bounds)
            self.differential_evolution(func, bounds)
        return self.global_best_position