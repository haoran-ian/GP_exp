import numpy as np

class DynamicPartitioningSwarmOptimizer:
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

    def dynamic_partitioning(self):
        sorted_indices = np.argsort(self.personal_best_scores)
        sub_swarm_size = self.particle_count // 2
        sub_swarms = [sorted_indices[:sub_swarm_size], sorted_indices[sub_swarm_size:]]
        return sub_swarms

    def phase_transition_detection(self):
        diversity = np.std(self.positions)
        phase_transition_threshold = 0.2 
        return diversity < phase_transition_threshold

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            sub_swarms = self.dynamic_partitioning()
            for sub_swarm in sub_swarms:
                if self.phase_transition_detection():
                    # Adjust parameters dynamically if in transition phase
                    self.inertia_weight_final = 0.7
                    self.c1_final = 1.8
                else:
                    self.inertia_weight_final = 0.4
                    self.c1_final = 1.5
                self.update_velocity_and_position()
        return self.global_best_position, self.global_best_score