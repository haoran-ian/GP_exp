import numpy as np

class Enhanced_Hybrid_PSO_MultiSwarm_v1:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40
        self.num_swarms = 3
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.exploration_weight = 0.5
        self.exploitation_weight = 0.3
        self.vel_decay = 0.95
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.swarm_best_positions = [None] * self.num_swarms
        self.swarm_best_values = [float('inf')] * self.num_swarms

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')
            self.swarm_id = None

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        # Distribute particles across swarms
        for idx, particle in enumerate(self.particles):
            particle.swarm_id = idx % self.num_swarms

        evaluations = 0
        while evaluations < self.budget:
            for particle in self.particles:
                fitness_value = func(particle.position)
                evaluations += 1

                if fitness_value < particle.best_value:
                    particle.best_value = fitness_value
                    particle.best_position = np.copy(particle.position)

                swarm_id = particle.swarm_id
                if fitness_value < self.swarm_best_values[swarm_id]:
                    self.swarm_best_values[swarm_id] = fitness_value
                    self.swarm_best_positions[swarm_id] = np.copy(particle.position)

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = np.copy(particle.position)

                if evaluations >= self.budget:
                    break

                # Adaptive inertia weight strategy
                self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)

                # Update particle velocity and position
                r1, r2, r3 = np.random.uniform(size=3)
                cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                swarm_social_velocity = self.c2 * r3 * (self.swarm_best_positions[swarm_id] - particle.position)
                particle.velocity = (self.exploration_weight * swarm_social_velocity +
                                     self.exploitation_weight * (cognitive_velocity + social_velocity) +
                                     self.w * particle.velocity)
                particle.position = particle.position + particle.velocity

                # Constrain to bounds
                particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                # Adaptive velocity tuning
                particle.velocity *= self.vel_decay

        return self.global_best_position, self.global_best_value