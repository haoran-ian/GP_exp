import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveHybridPSODEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.particles.copy()
        self.global_best_position = self.particles[0].copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.c1, self.c2 = 1.5, 2.0
        self.w = 0.5
        self.de_cross_rate = 0.8
        self.dynamic_topology = True
        self.local_chaotic_factor = 0.05
        self.diversity_threshold = 0.18
        self.neighborhood_size = 5
        self.evaluations = 0

    def update_velocity(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        if self.dynamic_topology:
            neighborhood_best = self.get_neighborhood_best()
            social_component = self.c2 * r2 * (neighborhood_best - self.particles)
        else:
            social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        self.velocities = np.clip(self.velocities, -0.1, 0.1)

    def get_neighborhood_best(self):
        neighborhood_best_positions = np.copy(self.particles)
        for i in range(self.population_size):
            neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
            neighborhood_best_value = np.inf
            for idx in neighbors_indices:
                if self.personal_best_values[idx] < neighborhood_best_value:
                    neighborhood_best_value = self.personal_best_values[idx]
                    neighborhood_best_positions[i] = self.personal_best_positions[idx]
        return neighborhood_best_positions

    def update_position(self, bounds):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, bounds.lb, bounds.ub)

    def chaotic_local_search(self, func, bounds):
        for i in range(self.population_size):
            perturbation = self.local_chaotic_factor * (np.random.rand(self.dim) - 0.5)
            candidate = self.particles[i] + perturbation
            candidate = np.clip(candidate, bounds.lb, bounds.ub)
            candidate_value = func(candidate)
            self.evaluations += 1
            if candidate_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = candidate
                self.personal_best_values[i] = candidate_value
                if candidate_value < self.global_best_value:
                    self.global_best_position = candidate
                    self.global_best_value = candidate_value
            if self.evaluations >= self.budget:
                return

    def measure_diversity(self):
        centroid = np.mean(self.particles, axis=0)
        diversity = np.mean(np.linalg.norm(self.particles - centroid, axis=1))
        return diversity

    def enhance_diversity(self, bounds):
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(self.particles)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                self.particles[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            else:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.particles[i] = 0.5 * (self.particles[i] + cluster_center)
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def update_dynamic_parameters(self):
        self.w = max(0.2, self.w * 0.98)
        self.c1 = max(1.0, self.c1 * 0.98)
        self.c2 = min(2.5, self.c2 * 1.02)

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            self.update_velocity()
            self.update_position(bounds)
            self.chaotic_local_search(func, bounds)
            if self.measure_diversity() < self.diversity_threshold:
                self.enhance_diversity(bounds)
            if self.dynamic_topology:
                self.update_dynamic_parameters()
        return self.global_best_position