import numpy as np

class Enhanced_MultiSwarm_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.swarm_size = 10
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.9   # initial inertia weight
        self.w_min = 0.4
        self.vel_decay = 0.95
        self.mutation_rate = 0.15
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
        def __init__(self, swarm_size, dim, bounds):
            self.particles = [Enhanced_MultiSwarm_PSO.Particle(dim, bounds) for _ in range(swarm_size)]
            self.local_best_position = None
            self.local_best_value = float('inf')

    def __call__(self, func):
        bounds = func.bounds
        self.swarms = [self.Swarm(self.swarm_size, self.dim, bounds) for _ in range(self.num_swarms)]

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

                    self.w = self.w_min + (0.5 * np.cos(np.pi * evaluations / self.budget))

                    if evaluations >= self.budget:
                        break

                    r1, r2 = np.random.uniform(size=2)
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (swarm.local_best_position - particle.position)
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity

                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)
                    particle.velocity *= self.vel_decay

                    current_mutation_rate = self.mutation_rate * (1 - evaluations / self.budget)
                    if np.random.rand() < current_mutation_rate:
                        mutation_vector = np.random.normal(0, 0.1, self.dim)
                        particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

        return self.global_best_position, self.global_best_value