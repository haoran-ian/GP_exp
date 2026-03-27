import numpy as np

class PSO_ALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1_start, self.c1_end = 2.5, 0.5  # dynamic cognitive coefficient
        self.c2_start, self.c2_end = 0.5, 2.5  # dynamic social coefficient
        self.w = 0.7   # inertia weight
        self.vel_decay = 0.99  # adaptive velocity decay
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []

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
            for particle in self.particles:
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

                # Dynamic learning rate adjustment
                t = evaluations / self.budget
                c1 = self.c1_start * (1 - t) + self.c1_end * t
                c2 = self.c2_start * (1 - t) + self.c2_end * t

                # Update particle velocity and position
                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
                social_velocity = c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

        return self.global_best_position, self.global_best_value