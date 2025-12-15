import numpy as np

class AdaptiveExploratorySwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(40, self.budget // 10)
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.c1 = 2.05  # cognitive component
        self.c2 = 2.05  # social component
        self.w = 0.5    # inertia weight
        self.bounds = None
        self.func_evals = 0

    def initialize(self, bounds):
        self.bounds = bounds
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.full(self.dim, (ub + lb) / 2)  # Initialized to mid-point of bounds

    def __call__(self, func):
        self.initialize(func.bounds)

        while self.func_evals < self.budget:
            for i in range(self.population_size):
                self.update_particle(i, func)

            self.update_parameters()

        return self.global_best_position, self.global_best_score

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(), np.random.rand()

        # Update velocity
        cognitive_component = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
        social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
        self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
        self.velocities[i] = np.clip(self.velocities[i], -0.1 * (self.bounds.ub - self.bounds.lb), 0.1 * (self.bounds.ub - self.bounds.lb))

        # Update position
        self.positions[i] = self.positions[i] + self.velocities[i]

        # Apply bounds
        self.positions[i] = np.clip(self.positions[i], self.bounds.lb, self.bounds.ub)

        # Evaluate fitness
        score = func(self.positions[i])
        self.func_evals += 1

        # Update personal best
        if score < self.best_scores[i]:
            self.best_scores[i] = score
            self.best_positions[i] = self.positions[i]

        # Update global best
        if score < self.global_best_score:
            self.global_best_score = score
            self.global_best_position = self.positions[i]

    def update_parameters(self):
        # Adapt parameters based on the progress
        if self.func_evals > self.budget / 2:
            self.c1 *= 0.98  # Reduced decay rate
            self.c2 *= 1.02  # Increased growth rate
        self.w = max(0.1, self.w * 0.99)