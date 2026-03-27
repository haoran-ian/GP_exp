import numpy as np

class Enhanced_Hybrid_PSO_ALS_v5:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40  # Increased number of particles
        self.c1 = 1.7  # Adjust cognitive coefficient
        self.c2 = 1.7  # Adjust social coefficient
        self.w_min = 0.3
        self.w_max = 0.8
        self.vel_decay = 0.95  # Adjust velocity decay
        self.mutation_rate = 0.2  # Increase mutation rate
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.restart_prob = 0.03  # Lower restart probability
        self.elite_fraction = 0.2  # New elite selection parameter
        self.cluster_distance = 0.1 * (dim ** 0.5)
        self.subpop_size_factor = 0.1

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.random.uniform(-0.1, 0.1, dim)  # Initialize with small random velocity
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        while evaluations < self.budget:
            self.subpop_size = max(2, int(self.num_particles * self.subpop_size_factor))
            subpop_indices = np.random.permutation(self.num_particles).reshape(-1, self.subpop_size)
            elite_count = int(self.num_particles * self.elite_fraction)
            elite_indices = np.argsort([p.best_value for p in self.particles])[:elite_count]

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

                    if evaluations >= self.budget:
                        break

                    self.w = self.w_max - (self.w_max - self.w_min) * ((evaluations + i) / self.budget)  # Adaptive inertia

                    r1, r2 = np.random.uniform(size=2)
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)

                    if i in elite_indices:
                        social_velocity *= 1.2  # Boost elite particles
                        
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity

                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                    particle.velocity *= self.vel_decay

                    if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                        if self.is_particle_isolated(particle):
                            mutation_vector = np.random.normal(0, 0.1, self.dim)
                            particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                    if np.random.rand() < self.restart_prob:
                        particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                        particle.velocity = np.random.uniform(-0.1, 0.1, self.dim)  # Restart with random velocity
                        particle.best_position = np.copy(particle.position)
                        particle.best_value = float('inf')

        return self.global_best_position, self.global_best_value

    def is_particle_isolated(self, particle):
        distances = [np.linalg.norm(particle.position - other.position) for other in self.particles if other != particle]
        return all(distance > self.cluster_distance for distance in distances)