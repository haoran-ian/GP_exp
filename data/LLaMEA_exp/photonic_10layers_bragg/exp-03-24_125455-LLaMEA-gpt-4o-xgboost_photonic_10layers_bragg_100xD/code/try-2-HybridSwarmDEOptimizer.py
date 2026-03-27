import numpy as np

class HybridSwarmDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 30
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.9
        self.inertia_decay = 0.97  # Adjusted inertia decay for faster adaptability
        self.c1 = 1.5  # Adjusted cognitive coefficient
        self.c2 = 1.7  # Adjusted social coefficient
        self.de_f = 0.5  # DE mutation factor
        self.de_cr = 0.9  # DE crossover probability
        self.eval_count = 0

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.particle_count, float('inf'))

    def differential_evolution(self, idx):
        candidates = np.random.choice(self.particle_count, 3, replace=False)
        a, b, c = self.positions[candidates]
        mutant_vector = a + self.de_f * (b - c)
        trial_vector = np.where(np.random.rand(self.dim) < self.de_cr, mutant_vector, self.positions[idx])
        return trial_vector

    def update_velocity_and_position(self):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        for i in range(self.particle_count):
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                                  self.c2 * r2 * (self.global_best_position - self.positions[i]))
            if np.random.rand() < 0.5:  # Hybrid strategy
                self.positions[i] = self.differential_evolution(i)
            else:
                self.positions[i] += self.velocities[i]
        self.inertia_weight *= self.inertia_decay

    def evaluate_particles(self, func):
        for i in range(self.particle_count):
            if self.eval_count < self.budget:
                score = func(self.positions[i])
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
        return self.global_best_position, self.global_best_score