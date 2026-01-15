import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # initial inertia weight
        self.w_max = 0.9
        self.w_min = 0.4
        self.vel_decay = 0.99
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.diversity_threshold = 1e-5

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def diversity(self):
        return np.mean([np.linalg.norm(p.position - self.global_best_position) for p in self.particles])

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        no_improvement_counter = 0
        while evaluations < self.budget:
            for particle in self.particles:
                fitness_value = func(particle.position)
                evaluations += 1

                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)
                    no_improvement_counter = 0

                self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)
                particle.velocity *= self.vel_decay

            if no_improvement_counter > 20:
                for particle in self.particles:
                    if np.random.rand() < 0.2:
                        particle.velocity += np.random.uniform(-0.1, 0.1, self.dim)
                no_improvement_counter = 0

            if self.diversity() < self.diversity_threshold:
                for particle in self.particles:
                    particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)

        return self.global_best_position, self.global_best_value