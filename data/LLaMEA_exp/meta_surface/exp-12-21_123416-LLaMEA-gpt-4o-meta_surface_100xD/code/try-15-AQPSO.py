import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.7
        self.social_coeff = 1.7
        self.adaptation_rate = 0.15
        self.q_factor = 0.5

    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))

    def update_particles(self, func):
        for i in range(self.pop_size):
            score = func(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]

        sorted_indices = np.argsort(self.personal_best_scores)
        elite_indices = sorted_indices[:int(self.pop_size * 0.15)]

        for i in range(self.pop_size):
            cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            quantum_component = np.random.uniform(-1, 1, self.dim) * self.q_factor * np.abs(self.global_best_position - self.positions[i])
            elite_component = np.mean([self.positions[idx] for idx in elite_indices], axis=0) - self.positions[i]
            
            self.velocities[i] = self.inertia_weight * self.velocities[i] \
                                 + cognitive_component \
                                 + social_component \
                                 + self.adaptation_rate * elite_component \
                                 + quantum_component
            
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0
        while evaluations < self.budget:
            self.update_particles(func)
            evaluations += self.pop_size
        return self.global_best_position, self.global_best_score