import numpy as np

class HierarchicalAdaptiveSwarmOptimizer:
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
        self.group_best_positions = None
        self.group_best_scores = None
        self.inertia_weight_initial = 0.9  
        self.inertia_weight_final = 0.4  
        self.c1_initial = 2.0  
        self.c2_initial = 2.0  
        self.c1_final = 1.5  
        self.c2_final = 1.5  
        self.eval_count = 0
        self.group_count = 5  # Number of groups for hierarchical strategy
        self.groups = [[] for _ in range(self.group_count)]

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.particle_count, float('inf'))
        self.group_best_positions = np.copy(self.positions[:self.group_count])
        self.group_best_scores = np.full(self.group_count, float('inf'))
        self.assign_particles_to_groups()

    def assign_particles_to_groups(self):
        np.random.shuffle(self.positions)
        group_size = self.particle_count // self.group_count
        for i in range(self.group_count):
            self.groups[i] = list(range(i * group_size, (i + 1) * group_size))

    def update_velocity_and_position(self):
        for group_id, group in enumerate(self.groups):
            r1, r2 = np.random.rand(len(group), self.dim), np.random.rand(len(group), self.dim)
            t = self.eval_count / self.budget
            inertia_weight = (1 - t) * self.inertia_weight_initial + t * self.inertia_weight_final
            c1 = (1 - t) * self.c1_initial + t * self.c1_final
            c2 = (1 - t) * self.c2_initial + t * self.c2_final

            for i in group:
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      c1 * r1[i % len(group)] * (self.personal_best_positions[i] - self.positions[i]) +
                                      c2 * r2[i % len(group)] * (self.group_best_positions[group_id] - self.positions[i]) +
                                      c2 * r2[i % len(group)] * (self.global_best_position - self.positions[i]))
                self.positions[i] += self.velocities[i]

    def evaluate_particles(self, func):
        for i in range(self.particle_count):
            if self.eval_count < self.budget:
                score = func(self.positions[i])
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                group_id = i // (self.particle_count // self.group_count)
                if score < self.group_best_scores[group_id]:
                    self.group_best_scores[group_id] = score
                    self.group_best_positions[group_id] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
        return self.global_best_position, self.global_best_score