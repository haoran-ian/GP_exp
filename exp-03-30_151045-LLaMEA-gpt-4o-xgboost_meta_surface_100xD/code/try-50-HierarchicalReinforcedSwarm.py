import numpy as np

class HierarchicalReinforcedSwarm:
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
        self.subgroup_factor = 0.5  # Fraction of population to form subgroups

    def _adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * (self.current_eval / self.budget))

    def _chaos_perturbation(self, position, bounds):
        beta = 0.3 * (1 - self.current_eval / self.budget)
        z = np.random.standard_cauchy(self.dim)
        chaotic_step = beta * z
        return position + chaotic_step

    def _dynamic_boundary_scaling(self, bounds):
        scaling_factor = 0.1 * (1 - self.current_eval / self.budget)
        return bounds.lb + scaling_factor * (bounds.ub - bounds.lb), bounds.ub - scaling_factor * (bounds.ub - bounds.lb)

    def _dynamic_population_size(self):
        return max(self.min_population_size, int(self.population_size * (1 - self.current_eval / self.budget)))

    def _hierarchical_clustering(self):
        # Implement hierarchical clustering mechanism
        cluster_centers = self.particles[np.random.choice(self.population_size, 5, replace=False)]
        clusters = {i: [] for i in range(len(cluster_centers))}
        for particle in self.particles:
            distances = np.linalg.norm(cluster_centers - particle, axis=1)
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(particle)
        # Further subdivide each cluster to improve focus on local optima
        subclusters = {}
        for cluster_idx, members in clusters.items():
            if len(members) > 1:
                subgroup_size = max(1, int(len(members) * self.subgroup_factor))
                subcluster_centers = np.random.choice(len(members), subgroup_size, replace=False)
                subclusters[cluster_idx] = {sc: [] for sc in subcluster_centers}
                for member in members:
                    sc_distances = np.linalg.norm(np.array([members[sc] for sc in subcluster_centers]) - member, axis=1)
                    closest_subcluster = np.argmin(sc_distances)
                    subclusters[cluster_idx][closest_subcluster].append(member)
        return clusters, subclusters

    def _reinforced_learning_rates(self):
        # Reinforce learning rates based on progress and cluster feedback
        progress = self.current_eval / self.budget
        self.c1 = self.c1_initial * (1 - progress)
        self.c2 = self.c2_initial * progress

    def __call__(self, func):
        bounds = func.bounds
        while self.current_eval < self.budget:
            lower_bound, upper_bound = self._dynamic_boundary_scaling(bounds)
            self.population_size = self._dynamic_population_size()
            self._reinforced_learning_rates()

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
            clusters, subclusters = self._hierarchical_clustering()

            for i in range(self.population_size):
                cluster_idx = np.argmin([np.linalg.norm(self.particles[i] - center) for center in clusters.keys()])
                cluster_center = np.mean(clusters[cluster_idx], axis=0)

                # Subgroup focus for enhancing local search
                subgroup_idx = np.argmin([np.linalg.norm(self.particles[i] - center) for center in subclusters[cluster_idx].keys()])
                subgroup_center = np.mean(subclusters[cluster_idx][subgroup_idx], axis=0)

                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      self.c1 * np.random.rand() * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * np.random.rand() * (self.global_best_position - subgroup_center))
                new_particle = self.particles[i] + self.velocities[i]
                new_particle = self._chaos_perturbation(new_particle, bounds)
                self.particles[i] = np.clip(new_particle, lower_bound, upper_bound)

        return self.global_best_position, self.global_best_value