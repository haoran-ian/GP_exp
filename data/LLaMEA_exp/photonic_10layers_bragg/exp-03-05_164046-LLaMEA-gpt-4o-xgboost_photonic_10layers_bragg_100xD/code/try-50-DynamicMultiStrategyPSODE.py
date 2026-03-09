import numpy as np
from sklearn.cluster import KMeans

class DynamicMultiStrategyPSODE:
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
        self.de_f = 0.5
        self.evaluations = 0
        self.diversity_threshold = 0.15
        self.temperature = 1.0
        self.cooling_rate = 0.99

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

    def simulated_annealing(self, func, bounds):
        for i in range(self.population_size):
            new_pos = self.particles[i] + self.temperature * np.random.uniform(-1, 1, self.dim)
            new_pos = np.clip(new_pos, bounds.lb, bounds.ub)
            new_val = func(new_pos)
            self.evaluations += 1
            if new_val < self.personal_best_values[i] or np.exp((self.personal_best_values[i] - new_val) / self.temperature) > np.random.rand():
                self.personal_best_positions[i] = new_pos
                self.personal_best_values[i] = new_val
                if new_val < self.global_best_value:
                    self.global_best_position = new_pos
                    self.global_best_value = new_val
            if self.evaluations >= self.budget:
                return

    def chaotic_map_guided_search(self, bounds):
        chaos_sequence = np.sin(np.arange(0, self.population_size) * 0.1)
        for i, chaos_val in enumerate(chaos_sequence):
            if np.random.rand() < abs(chaos_val):
                self.particles[i] = self.particles[i] + chaos_val * (self.global_best_position - self.particles[i])
                self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def update_dynamic_parameters(self):
        self.w = max(0.2, self.w * 0.98)
        self.c1 = max(1.0, self.c1 * 0.98)
        self.c2 = min(2.5, self.c2 * 1.02)
        self.temperature *= self.cooling_rate

    def __call__(self, func):
        bounds = func.bounds
        self.evaluations = 0
        while self.evaluations < self.budget:
            self.update_velocity()
            self.update_position(bounds)
            self.simulated_annealing(func, bounds)
            if self.measure_diversity() < self.diversity_threshold:
                self.chaotic_map_guided_search(bounds)
            self.update_dynamic_parameters()
        return self.global_best_position