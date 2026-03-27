import numpy as np

class Enhanced_Hybrid_PSO_ALS_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # initial inertia weight
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.vel_decay = 0.98  # adaptive velocity decay
        self.mutation_rate = 0.12  # mutation rate for exploration
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.restart_prob = 0.05  # probability of random restarts

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
                
                # Adjust inertia weight based on improvement
                self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

                if evaluations >= self.budget:
                    break

                # Update particle velocity and position with adaptive learning rates
                adaptive_c1 = self.c1 * (1 - evaluations / self.budget)
                adaptive_c2 = self.c2 * (1 + evaluations / self.budget)
                r1, r2 = np.random.uniform(size=2)
                cognitive_velocity = adaptive_c1 * r1 * (particle.best_position - particle.position)
                social_velocity = adaptive_c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

                # Dynamic mutation rate
                current_mutation_rate = self.mutation_rate * (1 - evaluations / self.budget)
                if np.random.rand() < current_mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                # Diversity-based regrouping mechanism
                if evaluations % (self.budget // 5) == 0:
                    diversity = np.mean([np.linalg.norm(particle.position - self.global_best_position) for particle in self.particles])
                    if diversity < 0.1 * np.linalg.norm(bounds.ub - bounds.lb):
                        for particle in self.particles:
                            particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                            particle.velocity = np.zeros(self.dim)
                            particle.best_position = np.copy(particle.position)
                            particle.best_value = float('inf')

                # Introduce random restarts to escape local optima
                if np.random.rand() < self.restart_prob:
                    particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    particle.velocity = np.zeros(self.dim)
                    particle.best_position = np.copy(particle.position)
                    particle.best_value = float('inf')

        return self.global_best_position, self.global_best_value