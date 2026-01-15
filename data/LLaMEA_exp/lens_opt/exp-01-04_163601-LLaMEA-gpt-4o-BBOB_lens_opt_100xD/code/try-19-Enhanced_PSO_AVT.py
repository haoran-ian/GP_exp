import numpy as np

class Enhanced_PSO_AVT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.initial_c1 = 2.5  # initial cognitive coefficient
        self.initial_c2 = 0.5  # initial social coefficient
        self.final_c1 = 0.5    # final cognitive coefficient
        self.final_c2 = 2.5    # final social coefficient
        self.w = 0.7           # initial inertia weight
        self.w_max = 0.9       # maximum inertia weight
        self.w_min = 0.4       # minimum inertia weight
        self.vel_decay = 0.99  # adaptive velocity decay
        self.global_best_position = None
        self.global_best_value = float('inf')

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
            c1 = self.initial_c1 - (self.initial_c1 - self.final_c1) * (evaluations / self.budget)
            c2 = self.initial_c2 + (self.final_c2 - self.initial_c2) * (evaluations / self.budget)

            for particle in self.particles:
                fitness_value = func(particle.position)
                evaluations += 1

                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)

                self.w = self.w_min + 0.5 * (self.w_max - self.w_min) * (1 + np.cos(np.pi * evaluations / self.budget))

                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
                social_velocity = c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                particle.velocity *= self.vel_decay

        return self.global_best_position, self.global_best_value