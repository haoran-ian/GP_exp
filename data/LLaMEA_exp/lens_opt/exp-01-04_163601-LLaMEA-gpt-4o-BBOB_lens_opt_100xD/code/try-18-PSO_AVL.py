import numpy as np

class PSO_AVL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.5  # initial cognitive coefficient
        self.c2 = 1.5  # initial social coefficient
        self.w = 0.7   # initial inertia weight
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
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
            self.local_search_rate = 0.05

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

                # Adjust inertia weight based on improvement
                self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
                
                # Adaptive cognitive and social coefficients
                self.c1 = 1.5 * (1 - (evaluations / self.budget))
                self.c2 = 1.5 * (evaluations / self.budget)

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

                # Local search refinement
                if np.random.rand() < particle.local_search_rate:
                    perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    new_position = particle.position + perturbation
                    new_position = np.clip(new_position, bounds.lb, bounds.ub)
                    new_fitness_value = func(new_position)
                    evaluations += 1

                    if new_fitness_value < fitness_value:
                        particle.position = new_position
                        particle.best_value = new_fitness_value
                        particle.best_position = np.copy(new_position)

                        if new_fitness_value < self.global_best_value:
                            self.global_best_value = new_fitness_value
                            self.global_best_position = np.copy(new_position)

        return self.global_best_position, self.global_best_value