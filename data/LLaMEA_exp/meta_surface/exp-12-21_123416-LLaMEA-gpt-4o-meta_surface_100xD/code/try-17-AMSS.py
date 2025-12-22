import numpy as np

class AMSS:
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
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.neighborhood_coeff = 0.3
        self.adaptive_rate = np.linspace(0.9, 0.4, self.budget // self.pop_size)
    
    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))

    def update_particles(self, func, step):
        # Adaptive inertia weight
        inertia_weight = self.adaptive_rate[step]
        
        for i in range(self.pop_size):
            score = func(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]

        # Sort particles for neighborhood influence
        sorted_indices = np.argsort(self.personal_best_scores)

        for i in range(self.pop_size):
            cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            
            # Adaptive neighborhood influence
            neighbor_index = sorted_indices[(i + 1) % self.pop_size]
            neighborhood_component = self.neighborhood_coeff * (self.personal_best_positions[neighbor_index] - self.positions[i])
            
            self.velocities[i] = inertia_weight * self.velocities[i] \
                                 + cognitive_component \
                                 + social_component \
                                 + neighborhood_component
            
            self.positions[i] += self.velocities[i]
            # Ensure particles remain within bounds
            self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0
        step = 0
        while evaluations < self.budget:
            self.update_particles(func, step)
            evaluations += self.pop_size
            step += 1
        return self.global_best_position, self.global_best_score