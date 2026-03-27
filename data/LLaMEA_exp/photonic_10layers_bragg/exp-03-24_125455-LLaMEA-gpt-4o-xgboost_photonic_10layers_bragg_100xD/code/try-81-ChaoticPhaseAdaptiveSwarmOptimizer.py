import numpy as np

class ChaoticPhaseAdaptiveSwarmOptimizer:
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
        self.chaotic_map_factor = np.random.rand(self.particle_count, self.dim)

    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.particle_count, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.particle_count, float('inf'))

    def chaotic_map_update(self):
        # Using Logistic map for chaotic sequence generation
        self.chaotic_map_factor = 4 * self.chaotic_map_factor * (1 - self.chaotic_map_factor)

    def update_velocity_and_position(self):
        r1, r2 = np.random.rand(self.particle_count, self.dim), np.random.rand(self.particle_count, self.dim)
        phase_factor = np.random.rand(self.particle_count, self.dim)
        t = self.eval_count / self.budget
        inertia_weight = (1 - t) * self.inertia_weight_initial + t * self.inertia_weight_final
        c1 = (1 - t) * self.c1_initial + t * self.c1_final
        c2 = (1 - t) * self.c2_initial + t * self.c2_final
        
        self.velocities = (inertia_weight * self.velocities +
                           c1 * r1 * (self.personal_best_positions - self.positions) +
                           c2 * r2 * (self.global_best_position - self.positions))
        self.positions += (1 - phase_factor * inertia_weight * self.chaotic_map_factor) * self.velocities

    def evaluate_particles(self, func):
        diversity_threshold = 0.1
        perturbation_scale = 1 - (self.eval_count / self.budget)
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
        
        if np.std(self.positions) < diversity_threshold:
            self.positions += np.random.uniform(-0.1, 0.1, self.positions.shape)

        if self.eval_count + self.particle_count <= self.budget:
            for i in range(self.particle_count):
                perturbation = np.random.uniform(-0.05, 0.05, self.dim) * perturbation_scale
                candidate_position = np.clip(self.positions[i] + perturbation, func.bounds.lb, func.bounds.ub)
                score = func(candidate_position)
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = candidate_position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = candidate_position

    def adaptive_neighborhood_search(self, func):
        for i in range(self.particle_count):
            neighborhood_radius = 0.05 * (func.bounds.ub - func.bounds.lb)
            local_best_score = float('inf')
            local_best_position = self.positions[i]
            for _ in range(5):
                neighbor_position = np.clip(self.positions[i] + np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim), func.bounds.lb, func.bounds.ub)
                score = func(neighbor_position)
                self.eval_count += 1
                if score < local_best_score:
                    local_best_score = score
                    local_best_position = neighbor_position
            if local_best_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = local_best_score
                self.personal_best_positions[i] = local_best_position
            if local_best_score < self.global_best_score:
                self.global_best_score = local_best_score
                self.global_best_position = local_best_position

    def dynamic_learning_strategy(self):
        if self.eval_count > self.budget * 0.5:
            self.c1_final *= 1.1
            self.c2_final *= 1.1

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
            self.adaptive_neighborhood_search(func)
            self.dynamic_learning_strategy()
            self.chaotic_map_update()
        return self.global_best_position, self.global_best_score