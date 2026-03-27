import numpy as np

class AdvancedAdaptiveSwarmOptimizer:
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
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.c1_final = 1.5
        self.c2_final = 1.5
        self.eval_count = 0

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.particle_count, float('inf'))

    def update_velocity_and_position(self):
        r1, r2 = np.random.rand(self.particle_count, self.dim), np.random.rand(self.particle_count, self.dim)
        t = self.eval_count / self.budget
        inertia_weight = (1 - t) * self.inertia_weight_initial + t * self.inertia_weight_final
        c1 = (1 - t) * self.c1_initial + t * self.c1_final
        c2 = (1 - t) * self.c2_initial + t * self.c2_final
        
        self.velocities = (inertia_weight * self.velocities +
                           c1 * r1 * (self.personal_best_positions - self.positions) +
                           c2 * r2 * (self.global_best_position - self.positions))
        self.positions += self.velocities

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

        diversity = np.std(self.positions)
        if diversity < 0.1:
            self.positions += np.random.uniform(-0.1, 0.1, self.positions.shape)

    def competitive_cooperative_learning(self, func):
        for i in range(self.particle_count):
            if i % 2 == 0:  # Cooperative learning
                neighbor_indices = np.random.choice(self.particle_count, 2, replace=False)
                neighbor_positions = self.positions[neighbor_indices]
                cooperative_position = np.mean(neighbor_positions, axis=0)
                score = func(cooperative_position)
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = cooperative_position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = cooperative_position
            else:  # Competitive learning
                for _ in range(3):
                    perturbation = np.random.uniform(-0.05, 0.05, self.dim)
                    competitor_position = np.clip(self.positions[i] + perturbation, func.bounds.lb, func.bounds.ub)
                    score = func(competitor_position)
                    self.eval_count += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = competitor_position
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = competitor_position

    def dynamic_learning_strategy(self):
        if self.eval_count > self.budget * 0.5:
            self.c1_final *= 1.1
            self.c2_final *= 1.1

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
            self.competitive_cooperative_learning(func)
            self.dynamic_learning_strategy()
        return self.global_best_position, self.global_best_score