import numpy as np

class PSO_AVT_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_particles = 30
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w_initial = 0.9  # initial inertia weight
        self.w_final = 0.4  # final inertia weight
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
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.initial_particles)]
        evaluations = 0

        while evaluations < self.budget:
            w = self.w_initial - (self.w_initial - self.w_final) * (evaluations / self.budget)
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

                # Update particle velocity and position
                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

            if evaluations % (self.budget // 5) == 0 and len(self.particles) > 10:
                self.particles = sorted(self.particles, key=lambda p: p.best_value)[:len(self.particles) - 5]

        return self.global_best_position, self.global_best_value