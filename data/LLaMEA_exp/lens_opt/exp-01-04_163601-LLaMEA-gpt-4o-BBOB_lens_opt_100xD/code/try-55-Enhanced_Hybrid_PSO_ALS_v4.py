import numpy as np

class Enhanced_Hybrid_PSO_ALS_v4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40  # increased number of particles for diversity
        self.c1 = 2.0  # higher cognitive coefficient for better individual exploration
        self.c2 = 2.0  # higher social coefficient for better convergence
        self.w = 0.9   # increased initial inertia weight
        self.w_max = 0.95  # increased maximum inertia weight
        self.w_min = 0.4  # unchanged minimum inertia weight
        self.vel_decay = 0.95  # slightly reduced velocity decay for more dynamic adjustment
        self.mutation_rate = 0.15  # increased mutation rate for more exploration
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.restart_prob = 0.04  # slightly reduced probability of random restarts
        self.cluster_distance = 0.1 * (dim ** 0.5)  # unchanged clustering threshold
        self.subpop_size = int(self.num_particles / 4)  # reduced size of subpopulations for more diverse grouping

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')
            self.history = []  # track historical best values

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        while evaluations < self.budget:
            subpop_indices = np.random.permutation(self.num_particles).reshape(-1, self.subpop_size)
            for indices in subpop_indices:
                for i in indices:
                    particle = self.particles[i]
                    fitness_value = func(particle.position)
                    evaluations += 1

                    if fitness_value < particle.best_value:
                        particle.best_value = fitness_value
                        particle.best_position = np.copy(particle.position)
                        particle.history.append(fitness_value)

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

                    # History-based mutation
                    if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                        if self.is_particle_isolated(particle):
                            history_std = np.std(particle.history[-min(len(particle.history), 5):]) if particle.history else 0.1
                            mutation_vector = np.random.normal(0, history_std, self.dim)
                            particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                    # Introduce random restarts
                    if np.random.rand() < self.restart_prob:
                        particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                        particle.velocity = np.zeros(self.dim)
                        particle.best_position = np.copy(particle.position)
                        particle.best_value = float('inf')
                        particle.history.clear()

        return self.global_best_position, self.global_best_value

    def is_particle_isolated(self, particle):
        distances = [np.linalg.norm(particle.position - other.position) for other in self.particles if other != particle]
        return all(distance > self.cluster_distance for distance in distances)