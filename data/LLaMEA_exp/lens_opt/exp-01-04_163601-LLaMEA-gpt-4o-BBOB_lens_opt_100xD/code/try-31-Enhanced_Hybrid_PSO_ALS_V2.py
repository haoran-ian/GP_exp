import numpy as np

class Enhanced_Hybrid_PSO_ALS_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.w_max = 0.9
        self.w_min = 0.4
        self.vel_decay = 0.98
        self.mutation_rate = 0.12
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.restart_prob = 0.05
        self.diversity_threshold = 1e-5  # diversity threshold for resampling

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def calculate_diversity(self):
        positions = np.array([p.position for p in self.particles])
        return np.mean(np.std(positions, axis=0))

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        while evaluations < self.budget:
            diversity = self.calculate_diversity()

            for particle in self.particles:
                fitness_value = func(particle.position)
                evaluations += 1

                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)
                
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

                current_mutation_rate = self.mutation_rate * (1 - evaluations / self.budget)
                if np.random.rand() < current_mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    particle.position = np.clip(particle.position + mutation_vector, bounds.lb, bounds.ub)

                if np.random.rand() < self.restart_prob:
                    particle.position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    particle.velocity = np.zeros(self.dim)
                    particle.best_position = np.copy(particle.position)
                    particle.best_value = float('inf')

            # Stochastic swarm resampling if diversity is too low
            if diversity < self.diversity_threshold:
                resample_idx = np.random.choice(self.num_particles, size=self.num_particles // 2, replace=False)
                for idx in resample_idx:
                    self.particles[idx].position = np.random.uniform(bounds.lb, bounds.ub, self.dim)
                    self.particles[idx].velocity = np.zeros(self.dim)
                    self.particles[idx].best_position = np.copy(self.particles[idx].position)
                    self.particles[idx].best_value = float('inf')

        return self.global_best_position, self.global_best_value