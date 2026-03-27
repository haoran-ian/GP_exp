import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_num_particles = 30
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.vel_decay = 0.99  # adaptive velocity decay
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.evaluations = 0

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.random.uniform(-1, 1, dim) * (bounds.ub - bounds.lb) * 0.1
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def adjust_population_size(self):
        # Dynamically adjust number of particles based on evaluation progress
        progress = self.evaluations / self.budget
        num_particles = int(self.initial_num_particles * (1 + (1 - progress) * 0.5))
        return max(5, num_particles)

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.initial_num_particles)]

        while self.evaluations < self.budget:
            num_particles = self.adjust_population_size()
            if len(self.particles) != num_particles:
                self.particles = self.particles[:num_particles] + [
                    self.Particle(self.dim, bounds) for _ in range(num_particles - len(self.particles))
                ]

            for particle in self.particles:
                if self.evaluations >= self.budget:
                    break

                fitness_value = func(particle.position)
                self.evaluations += 1

                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)
                
                # Adjust inertia weight based on improvement
                self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

                # Update particle velocity and position
                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity

                # Implement velocity scaling to avoid explosion
                velocity_norm = np.linalg.norm(particle.velocity)
                max_velocity = (bounds.ub - bounds.lb) * 0.1
                if velocity_norm > max_velocity:
                    particle.velocity = particle.velocity / velocity_norm * max_velocity

                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

        return self.global_best_position, self.global_best_value