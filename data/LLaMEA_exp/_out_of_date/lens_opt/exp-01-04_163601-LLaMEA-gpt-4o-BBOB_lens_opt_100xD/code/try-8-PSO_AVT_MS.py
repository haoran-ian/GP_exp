import numpy as np

class PSO_AVT_MS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.num_particles_per_swarm = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.vel_decay = 0.99
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.swarms = []

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    class Swarm:
        def __init__(self, dim, num_particles, bounds):
            self.particles = [PSO_AVT_MS.Particle(dim, bounds) for _ in range(num_particles)]
            self.local_best_position = None
            self.local_best_value = float('inf')

    def __call__(self, func):
        bounds = func.bounds
        self.swarms = [self.Swarm(self.dim, self.num_particles_per_swarm, bounds) for _ in range(self.num_swarms)]

        evaluations = 0
        while evaluations < self.budget:
            for swarm in self.swarms:
                for particle in swarm.particles:
                    fitness_value = func(particle.position)
                    evaluations += 1

                    if fitness_value < particle.best_value:
                        particle.best_value = fitness_value
                        particle.best_position = np.copy(particle.position)

                    if fitness_value < swarm.local_best_value:
                        swarm.local_best_value = fitness_value
                        swarm.local_best_position = np.copy(particle.position)

                    if fitness_value < self.global_best_value:
                        self.global_best_value = fitness_value
                        self.global_best_position = np.copy(particle.position)

                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.uniform(size=2)
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (swarm.local_best_position - particle.position)
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position += particle.velocity

                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)
                    particle.velocity *= self.vel_decay

                # Update swarm's local best with global best occasionally
                if np.random.rand() < 0.1:  # 10% chance of updating with global best
                    if self.global_best_value < swarm.local_best_value:
                        swarm.local_best_value = self.global_best_value
                        swarm.local_best_position = np.copy(self.global_best_position)

        return self.global_best_position, self.global_best_value