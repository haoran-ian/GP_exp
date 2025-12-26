import numpy as np

class EnhancedNAPSOLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(100, int(np.sqrt(budget)))
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.neighbors = 5  # Number of neighbors for neighborhood-based learning

    def initialize(self, lb, ub):
        self.particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / abs(v)**(1 / beta)
        return step

    def update_particles(self, lb, ub, evaluations):
        w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))
        adapt_c1 = self.c1 * (1 - evaluations / self.budget)
        for i in range(self.num_particles):
            # Select a random subset of neighbors
            neighbor_indices = np.random.choice(self.num_particles, self.neighbors, replace=False)
            local_best_index = neighbor_indices[np.argmin(self.personal_best_scores[neighbor_indices])]
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities[i] = (w * self.velocities[i] +
                                  adapt_c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                  self.c2 * r2 * (self.personal_best_positions[local_best_index] - self.particles[i]))
            
            # Adding stochastic variability to velocity
            self.velocities[i] += np.random.normal(0, 0.1, self.dim)
            
            # Incorporating Lévy flight
            if np.random.rand() < 0.1:
                self.velocities[i] += self.levy_flight(self.dim)
            
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

            self.update_particles(lower_bound, upper_bound, evaluations)

        return self.global_best_position, self.global_best_score