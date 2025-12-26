import numpy as np

class QuantumEnhancedAdaptiveSwarmOptimizerWithMutation:
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
        self.mutation_rate = 0.1
        self.quantum_tunnel_prob = 0.05

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
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)
            
            # Mutation
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.normal(0, 0.1 * (ub - lb), self.dim)
                self.positions[i] = np.clip(self.positions[i] + mutation_vector, lb, ub)
            
            # Quantum tunneling
            if np.random.rand() < self.quantum_tunnel_prob:
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
            self.cognitive_const = 1.0 + (0.4 * (self.budget - evaluations) / self.budget)
            self.social_const = 2.0 - (0.4 * (self.budget - evaluations) / self.budget)
            self.update_positions_and_velocities(bounds)
            self.evaluate_and_update_best(func)
            evaluations += self.population_size
            self.inertia_weight = max(0.4, self.inertia_weight - self.inertia_decay)

        return self.global_best_position