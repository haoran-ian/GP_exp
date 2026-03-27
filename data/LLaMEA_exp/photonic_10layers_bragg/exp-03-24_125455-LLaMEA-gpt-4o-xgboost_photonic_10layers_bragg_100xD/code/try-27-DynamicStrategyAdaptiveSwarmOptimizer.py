import numpy as np
from sklearn.cluster import KMeans

class DynamicStrategyAdaptiveSwarmOptimizer:
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

    def cluster_based_exploration(self, func):
        if self.eval_count < self.budget * 0.8:
            kmeans = KMeans(n_clusters=min(self.particle_count // 5, self.particle_count), random_state=0).fit(self.positions)
            for i in range(self.particle_count):
                if self.eval_count < self.budget:
                    cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                    candidate_position = np.clip(cluster_center + perturbation, func.bounds.lb, func.bounds.ub)
                    score = func(candidate_position)
                    self.eval_count += 1
                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = candidate_position
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = candidate_position

    def adaptive_parameters(self, func):
        diversity = np.linalg.norm(np.std(self.positions, axis=0))
        self.c1_initial = max(1.5, 3.0 - 1.5 * diversity)
        self.c2_initial = max(1.5, 3.0 - 1.5 * diversity)

    def __call__(self, func):
        self.initialize_particles(func.bounds)
        while self.eval_count < self.budget:
            self.evaluate_particles(func)
            self.update_velocity_and_position()
            self.cluster_based_exploration(func)
            self.adaptive_parameters(func)
        return self.global_best_position, self.global_best_score