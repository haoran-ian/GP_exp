import numpy as np

class EnhancedQuantumLevySwarmOptimizerV6:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.9
        self.cognitive_const = 1.4
        self.social_const = 1.6
        self.velocity_clamp_factor = 0.1

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta *
                  2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(size) * sigma
        v = np.random.randn(size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def initialize(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.random.uniform(lb, ub, self.dim)

    def update_positions_and_velocities(self, bounds, progress_factor):
        lb, ub = bounds.lb, bounds.ub
        inertia_scale = 1 - progress_factor  
        diversity = np.std(self.positions, axis=0).mean()
        mutation_strength = 0.1 * (1 - progress_factor) * diversity

        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_const * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = self.social_const * r2 * (self.global_best_position - self.positions[i])
            levy_steps = self.levy_flight(self.dim)
            random_walk = np.random.normal(0, 1, self.dim) * mutation_strength

            self.velocities[i] = (inertia_scale * self.velocities[i] +
                                  cognitive_velocity + social_velocity +
                                  0.2 * levy_steps +
                                  random_walk)

            max_velocity = (0.15 - 0.05 * progress_factor) * (ub - lb)
            self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

            if np.random.rand() < 0.02:
                self.positions[i] = np.random.uniform(lb, ub, self.dim)

    def evaluate_and_update_best(self, func):
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0

        while evaluations < self.budget:
            progress_factor = evaluations / self.budget
            if progress_factor < 0.5:
                self.cognitive_const = 1.5
                self.social_const = 1.5
            else:
                self.cognitive_const = 1.2
                self.social_const = 1.8

            self.update_positions_and_velocities(bounds, progress_factor)
            self.evaluate_and_update_best(func)
            evaluations += self.population_size

        return self.global_best_position