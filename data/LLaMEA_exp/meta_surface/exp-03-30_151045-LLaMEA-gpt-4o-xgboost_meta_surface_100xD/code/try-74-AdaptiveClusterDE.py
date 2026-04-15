import numpy as np

class AdaptiveClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 60
        self.particles = np.random.uniform(0, 1, (self.population_size, dim))
        self.velocities = np.random.randn(self.population_size, dim)
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.current_eval = 0
        self.F = 0.8
        self.CR = 0.9
        self.cluster_centers = None

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _dynamic_mutation(self, population, index):
        idxs = [idx for idx in range(self.population_size) if idx != index]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        scaling_factor = self.F * (1 + 0.5 * np.sin(3 * np.pi * self.current_eval / self.budget))
        mutant = a + scaling_factor * (b - c)
        return mutant

    def _cluster_particles(self):
        from sklearn.cluster import KMeans
        num_clusters = max(2, int(self.population_size * 0.1))
        kmeans = KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(self.particles)
        self.cluster_centers = kmeans.cluster_centers_
        return labels

    def _update_velocities_and_positions(self, bounds, inertia_weight):
        for i in range(self.population_size):
            distances = np.linalg.norm(self.cluster_centers - self.particles[i], axis=1)
            closest_center = self.cluster_centers[np.argmin(distances)]
            self.velocities[i] = (inertia_weight * self.velocities[i]
                                  + self.c1 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
                                  + self.c2 * np.random.rand(self.dim) * (self.global_best_position - closest_center))
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            labels = self._cluster_particles()
            inertia_weight = self._adaptive_inertia_weight()

            for i in range(self.population_size):
                value = func(self.particles[i])
                self.current_eval += 1
                if value < self.personal_best_values[i]:
                    self.personal_best_values[i] = value
                    self.personal_best_positions[i] = self.particles[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.particles[i]

            for i in range(self.population_size):
                mutant = self._dynamic_mutation(self.particles, i)
                crossover = np.random.rand(self.dim) < self.CR
                new_particle = np.where(crossover, mutant, self.particles[i])
                self._update_velocities_and_positions(bounds, inertia_weight)

        return self.global_best_position, self.global_best_value