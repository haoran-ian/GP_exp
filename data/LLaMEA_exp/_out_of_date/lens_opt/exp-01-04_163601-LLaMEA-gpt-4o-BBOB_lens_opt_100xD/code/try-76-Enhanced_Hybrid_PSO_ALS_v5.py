import numpy as np

class Enhanced_Hybrid_PSO_ALS_v5:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.w_max = 0.9
        self.w_min = 0.4
        self.vel_decay = 0.97
        self.mutation_rate = 0.15
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.restart_prob = 0.05
        self.cluster_distance = 0.1 * (dim ** 0.5)
        self.subpop_size_factor = 0.1
        self.local_search_radius = 0.05 * (dim ** 0.5)

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        while evaluations < self.budget:
            # Adaptive subpopulation size
            self.subpop_size = max(2, int(self.num_particles * self.subpop_size_factor))
            subpop_indices = np.random.permutation(self.num_particles).reshape(-1, self.subpop_size)
            for indices in subpop_indices:
                for i in indices:
                    particle = self.particles[i]
                    fitness_value = func(particle.position)
                    evaluations += 1

                    if fitness_value < particle.best_value:
                        particle.best_value = fitness_value
                        particle.best_position = np.copy(particle.position)

                    if fitness_value < self.global_best_value:
                        self.global_best_value = fitness_value
                        self.global_best_position = np.copy(particle.position)
                    
                    # Dynamic inertia weight adjustment
                    self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

                    if evaluations >= self.budget:
                        break

                    # Update particle velocity and position
                    r1, r2 = np.random.uniform(size=2)
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity

                    # Constrain to bounds
                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                    # Adaptive velocity tuning
                    particle.velocity *= self.vel_decay

                    # Cluster-based mutation
                    if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                        if self.is_particle_isolated(particle):
                            mutation_vector = np.random.normal(0, 0.1, self.dim)
                            particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                    # Apply local search around promising particles
                    if np.random.rand() < 0.1:  # 10% chance to perform local search
                        self.local_search(particle, func, bounds, evaluations)

                    # Introduce random restarts
                    if np.random.rand() < self.restart_prob:
                        particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                        particle.velocity = np.zeros(self.dim)
                        particle.best_position = np.copy(particle.position)
                        particle.best_value = float('inf')

        return self.global_best_position, self.global_best_value

    def local_search(self, particle, func, bounds, evaluations):
        neighborhood_center = particle.position
        num_local_samples = 5  # Number of samples in local search
        for _ in range(num_local_samples):
            if evaluations >= self.budget:
                break
            local_sample = neighborhood_center + np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
            local_sample = np.clip(local_sample, bounds.lb, bounds.ub)
            local_fitness = func(local_sample)
            evaluations += 1
            if local_fitness < particle.best_value:
                particle.best_value = local_fitness
                particle.best_position = np.copy(local_sample)
                if local_fitness < self.global_best_value:
                    self.global_best_value = local_fitness
                    self.global_best_position = np.copy(local_sample)

    def is_particle_isolated(self, particle):
        distances = [np.linalg.norm(particle.position - other.position) for other in self.particles if other != particle]
        return all(distance > self.cluster_distance for distance in distances)