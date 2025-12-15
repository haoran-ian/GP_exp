import numpy as np

class EnhancedAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(max(dim * 3, 30), self.budget // 10)
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 2.05  # cognitive component
        self.c2 = 2.05  # social component
        self.w = 0.9  # initial inertia weight
        self.bounds = None
        self.func_evals = 0

    def initialize(self, bounds):
        self.bounds = bounds
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.full(self.dim, (ub + lb) / 2)

    def __call__(self, func):
        self.initialize(func.bounds)

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                self.update_particle(i, func)

            self.update_parameters()

        return self.global_best_position, self.global_best_score

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(), np.random.rand()

        # Dynamic neighborhood based on current best scores
        neighborhood_size = max(5, self.population_size // 5)
        neighborhood_indices = np.argsort(self.best_scores)[:neighborhood_size]
        neighborhood_best_position = self.best_positions[neighborhood_indices[np.argmin(self.best_scores[neighborhood_indices])]]

        cognitive_component = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
        social_component = self.c2 * r2 * (neighborhood_best_position - self.positions[i])

        # Adaptive velocity control based on progress
        progress_factor = 1 - self.func_evals / self.budget
        velocity_clamp = 0.15 * (self.bounds.ub - self.bounds.lb) * progress_factor
        self.velocities[i] = (self.w * np.random.uniform(0.5, 0.9) * self.velocities[i]
                             + cognitive_component + social_component)
        self.velocities[i] = np.clip(self.velocities[i], -velocity_clamp, velocity_clamp)

        self.positions[i] = self.positions[i] + self.velocities[i]
        self.positions[i] = np.clip(self.positions[i], self.bounds.lb, self.bounds.ub)

        score = func(self.positions[i])
        self.func_evals += 1

        if score < self.best_scores[i]:
            self.best_scores[i] = score
            self.best_positions[i] = self.positions[i]

        if score < self.global_best_score:
            self.global_best_score = score
            self.global_best_position = self.positions[i]

    def update_parameters(self):
        progress = self.func_evals / self.budget
        self.c1 = 2.5 - 1.5 * progress
        self.c2 = 1.5 + 2.0 * progress
        self.w = 0.9 - 0.5 * progress  # Slight adjustment to the inertia weight