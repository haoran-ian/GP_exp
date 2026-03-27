import numpy as np

class PSO_DNALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1_initial = 1.0  # initial cognitive coefficient
        self.c1_final = 2.0    # final cognitive coefficient
        self.c2_initial = 1.0  # initial social coefficient
        self.c2_final = 2.5    # final social coefficient
        self.w = 0.7           # inertia weight
        self.vel_decay = 0.99  # adaptive velocity decay
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.neighborhood_size = 3  # dynamic neighborhood size

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
            for i, particle in enumerate(self.particles):
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

                # Update coefficients dynamically
                c1 = self.c1_initial + (self.c1_final - self.c1_initial) * (evaluations / self.budget)
                c2 = self.c2_initial + (self.c2_final - self.c2_initial) * (evaluations / self.budget)

                # Determine neighborhood best
                neighbors = self.particles[max(0, i-self.neighborhood_size):min(self.num_particles, i+self.neighborhood_size+1)]
                neighborhood_best = min(neighbors, key=lambda x: x.best_value).best_position

                # Update particle velocity and position
                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
                social_velocity = c2 * r2 * (neighborhood_best - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

        return self.global_best_position, self.global_best_value