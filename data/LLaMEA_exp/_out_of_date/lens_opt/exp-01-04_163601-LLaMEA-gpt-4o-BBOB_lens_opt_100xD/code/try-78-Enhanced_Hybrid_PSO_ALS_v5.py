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
        adaptive_rate = lambda evals: max(0.05, 0.5 - evals / (2 * self.budget))
        while evaluations < self.budget:
            self.subpop_size = max(2, int(self.num_particles * self.subpop_size_factor))
            subpop_indices = np.random.permutation(self.num_particles).reshape(-1, self.subpop_size)
            for indices in subpop_indices:
                leaders = sorted(indices, key=lambda i: self.particles[i].best_value)[:2]
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

                    self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.uniform(size=2)
                    leader_position = self.particles[np.random.choice(leaders)].best_position
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (leader_position - particle.position)
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity

                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)
                    particle.velocity *= self.vel_decay

                    if np.random.rand() < adaptive_rate(evaluations):
                        if self.is_particle_isolated(particle):
                            mutation_vector = np.random.normal(0, 0.1, self.dim)
                            particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                    if np.random.rand() < self.restart_prob:
                        particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                        particle.velocity = np.zeros(self.dim)
                        particle.best_position = np.copy(particle.position)
                        particle.best_value = float('inf')

        return self.global_best_position, self.global_best_value

    def is_particle_isolated(self, particle):
        distances = [np.linalg.norm(particle.position - other.position) for other in self.particles if other != particle]
        return all(distance > self.cluster_distance for distance in distances)