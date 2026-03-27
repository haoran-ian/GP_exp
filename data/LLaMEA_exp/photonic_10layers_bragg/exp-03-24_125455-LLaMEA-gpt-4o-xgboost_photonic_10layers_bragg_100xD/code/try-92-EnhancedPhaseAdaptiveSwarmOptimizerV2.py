import numpy as np

class EnhancedPhaseAdaptiveSwarmOptimizerV2:
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

    def phase_dynamic_perturbation(self, func):
        phase_factor = np.random.rand(self.particle_count, self.dim)
        for i in range(self.particle_count):
            if np.std(self.positions) < 0.1:
                perturbation = np.random.uniform(-0.05, 0.05, self.dim) * (1 - phase_factor)
                candidate_position = np.clip(self.positions[i] + perturbation, func.bounds.lb, func.bounds.ub)
                score = func(candidate_position)
                self.eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = candidate_position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = candidate_position

    def adaptive_memory_update(self):
        if self.eval_count > self.budget * 0.3:
            self.c1_final *= 1.05
            self.c2_final *= 1.05
        elif self.eval_count > self.budget * 0.7:
            self.c1_final *= 0.95
            self.c2_final *= 0.95

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
            self.phase_dynamic_perturbation(func)
            self.adaptive_memory_update()
        return self.global_best_position, self.global_best_score