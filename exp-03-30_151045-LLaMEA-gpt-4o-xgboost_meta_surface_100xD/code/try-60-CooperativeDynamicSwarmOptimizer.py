import numpy as np

class CooperativeDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.randn(self.population_size, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0
        self.min_population_size = 20

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _dynamic_population_size(self):
        return max(self.min_population_size, int(self.population_size * (1 - self.current_eval / self.budget)))

    def _adjust_learning_rates(self):
        progress = self.current_eval / self.budget
        self.c1 = self.c1_initial * (1 - progress)
        self.c2 = self.c2_initial * progress

    def _find_local_best(self, clusters):
        local_best_positions = np.zeros((len(clusters), self.dim))
        local_best_values = np.full(len(clusters), np.inf)
        for i, cluster in clusters.items():
            for particle in cluster:
                value = func(particle)
                if value < local_best_values[i]:
                    local_best_values[i] = value
                    local_best_positions[i] = particle
        return local_best_positions, local_best_values

    def _cluster_particles(self):
        cluster_centers = self.particles[np.random.choice(self.population_size, 5, replace=False)]
        clusters = {i: [] for i in range(len(cluster_centers))}
        for particle in self.particles:
            distances = np.linalg.norm(cluster_centers - particle, axis=1)
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(particle)
        return clusters

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            lower_bound, upper_bound = bounds.lb, bounds.ub
            self.population_size = self._dynamic_population_size()
            self._adjust_learning_rates()

            for i in range(self.population_size):
                value = func(self.particles[i])
                self.current_eval += 1

                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]

            inertia_weight = self._adaptive_inertia_weight()
            clusters = self._cluster_particles()
            local_best_positions, local_best_values = self._find_local_best(clusters)

            for i in range(self.population_size):
                cluster_idx = np.argmin([np.linalg.norm(self.particles[i] - center) for center in local_best_positions])
                local_best_position = local_best_positions[cluster_idx]

                self.velocities[i] = (
                    inertia_weight * self.velocities[i] +
                    self.c1 * np.random.rand() * (self.personal_best_positions[i] - self.particles[i]) +
                    self.c2 * np.random.rand() * (self.global_best_position - local_best_position)
                )
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

        return self.global_best_position, self.global_best_value