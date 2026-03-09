import numpy as np
from sklearn.cluster import KMeans

class RefinedAdaptiveHybridPSODEAlgorithm:
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
        self.w = 0.7
        self.de_cross_rate = 0.8
        self.de_f = 0.5
        self.evaluations = 0
        self.diversity_threshold = 0.15
        self.phase_length = self.budget // 4
        self.adaptive_learning_rate = 0.1
        self.dynamic_phase = 0

    def update_velocity(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.particles)
        social_component = self.c2 * r2 * (self.global_best_position - self.particles)
        self.velocities = self.w * self.velocities + cognitive_component + social_component
        self.velocities = np.clip(self.velocities, -0.1, 0.1)

    def update_position(self, bounds):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, bounds.lb, bounds.ub)

    def measure_diversity(self):
        centroid = np.mean(self.particles, axis=0)
        diversity = np.mean(np.linalg.norm(self.particles - centroid, axis=1))
        return diversity

    def differential_evolution(self, func, bounds):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
            de_factor = np.random.uniform(0.4, 0.9)
            mutant_vector = np.clip(a + de_factor * (b - c), bounds.lb, bounds.ub)
            crossover = np.random.rand(self.dim) < self.de_cross_rate
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_value = func(trial_vector)
            self.evaluations += 1
            if trial_value < self.personal_best_values[i]:
                self.personal_best_positions[i] = trial_vector
                self.personal_best_values[i] = trial_value
                if trial_value < self.global_best_value:
                    self.global_best_position = trial_vector
                    self.global_best_value = trial_value
            if self.evaluations >= self.budget:
                return

    def local_search(self, func, bounds):
        perturbation = np.random.normal(0, 0.005, size=self.dim)
        for i in range(self.population_size):
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

    def enhance_diversity(self, func, bounds):
        kmeans = KMeans(n_clusters=5 if self.dynamic_phase == 0 else 3)
        kmeans.fit(self.particles)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:
                self.particles[i] = np.random.uniform(bounds.lb, bounds.ub, self.dim)
            else:
                cluster_center = kmeans.cluster_centers_[kmeans.labels_[i]]
                self.particles[i] = 0.5 * (self.particles[i] + cluster_center)
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def update_dynamic_parameters(self):
        self.w = max(0.3, self.w * 0.98)
        self.c1 = max(1.0, self.c1 * 0.98)
        self.c2 = min(2.5, self.c2 * 1.02)
        self.adaptive_learning_rate *= 0.99
        self.dynamic_phase = (self.dynamic_phase + 1) % 2

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            self.update_velocity()
            self.update_position(bounds)
            self.differential_evolution(func, bounds)
            self.local_search(func, bounds)
            if self.measure_diversity() < self.diversity_threshold:
                self.enhance_diversity(func, bounds)
            self.update_dynamic_parameters()
        return self.global_best_position