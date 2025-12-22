import numpy as np

class MultiPhaseAdaptiveSwarm:
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
        self.c1_start = 2.0
        self.c2_start = 2.0
        self.c1_end = 1.5
        self.c2_end = 2.5
        self.w_start = 0.9
        self.w_end = 0.4
        self.bounds = None
        self.func_evals = 0

    def initialize(self, bounds):
        self.bounds = bounds
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.full(self.dim, (ub + lb) / 2)

    def __call__(self, func):
        self.initialize(func.bounds)

        while self.func_evals < self.budget:
            phase = self.func_evals / self.budget
            for i in range(self.population_size):
                self.update_particle(i, func, phase)

            self.update_parameters(phase)

        return self.global_best_position, self.global_best_score

    def update_particle(self, i, func, phase):
        r1, r2 = np.random.rand(), np.random.rand()

        cognitive_component = self.c1(phase) * r1 * (self.best_positions[i] - self.positions[i])
        social_component = self.c2(phase) * r2 * (self.global_best_position - self.positions[i])
        self.velocities[i] = self.w(phase) * self.velocities[i] + cognitive_component + social_component
        self.velocities[i] = np.clip(self.velocities[i], -0.1 * (self.bounds.ub - self.bounds.lb), 0.1 * (self.bounds.ub - self.bounds.lb))

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

    def update_parameters(self, phase):
        self.c1 = lambda p: (1-p)*self.c1_start + p*self.c1_end
        self.c2 = lambda p: (1-p)*self.c2_start + p*self.c2_end
        self.w = lambda p: (1-p)*self.w_start + p*self.w_end