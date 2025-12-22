import numpy as np

class NAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(100, int(np.sqrt(budget)))  # Dynamic choice of particles
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.5  # Inertia weight
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')

    def initialize(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))

    def update_particles(self, lb, ub):
        r1, r2 = np.random.rand(), np.random.rand()
        for i in range(self.num_particles):
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.global_best_position - self.particles[i]))

            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        lower_bound = np.array(func.bounds.lb)
        upper_bound = np.array(func.bounds.ub)
        self.initialize(lower_bound, upper_bound)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.particles[i])

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.particles[i])

                if evaluations >= self.budget:
                    break

            self.update_particles(lower_bound, upper_bound)

        return self.global_best_position, self.global_best_score