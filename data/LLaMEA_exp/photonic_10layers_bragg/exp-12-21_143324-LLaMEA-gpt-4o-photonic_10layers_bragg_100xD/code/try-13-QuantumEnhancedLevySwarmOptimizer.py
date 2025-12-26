import numpy as np

class QuantumEnhancedLevySwarmOptimizer:
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
        self.inertia_decay = (0.45 / self.budget) * self.population_size

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
        center = (lb + ub) / 2
        quantum_range = (ub - lb) / 2
        self.positions = np.random.uniform(center - quantum_range, center + quantum_range, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-quantum_range, quantum_range, (self.population_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = np.random.uniform(lb, ub, self.dim)

    def update_positions_and_velocities(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = self.cognitive_const * r1 * (self.best_positions[i] - self.positions[i])
            social_velocity = self.social_const * r2 * (self.global_best_position - self.positions[i])
            levy_vel = self.levy_flight(self.dim)
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_velocity + social_velocity +
                                  0.01 * levy_vel)
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

    def evaluate_and_update_best(self, func):
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = self.positions[i].copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i].copy()

    def dynamic_quantum_range(self, evaluations, total_budget, lb, ub):
        factor = 1 - (evaluations / total_budget)
        return ((ub - lb) / 2) * factor

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.cognitive_const = 1.0 + (0.4 * (self.budget - evaluations) / self.budget)
            self.social_const = 2.0 - (0.4 * (self.budget - evaluations) / self.budget)
            quantum_range = self.dynamic_quantum_range(evaluations, self.budget, bounds.lb, bounds.ub)
            self.positions = np.clip(self.positions + quantum_range * np.random.randn(self.population_size, self.dim), bounds.lb, bounds.ub)
            self.update_positions_and_velocities(bounds)
            self.evaluate_and_update_best(func)
            evaluations += self.population_size
            self.inertia_weight = max(0.4, self.inertia_weight - self.inertia_decay)

        return self.global_best_position