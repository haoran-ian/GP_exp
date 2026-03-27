import numpy as np

class EnhancedAdaptiveLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.w = 0.7  # increased inertia weight for better exploration
        self.c1 = 1.5  # decreased cognitive component for improved social learning
        self.c2 = 2.5  # increased social component for stronger convergence
        self.positions = self.initialize_positions()  # use chaos-inspired initialization
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def initialize_positions(self):
        # Chaos inspired initialization using logistic map
        positions = np.zeros((self.num_particles, self.dim))
        x = 0.7  # Initial condition for logistic map
        for i in range(self.num_particles):
            for j in range(self.dim):
                x = 3.9 * x * (1 - x)  # Logistic map equation
                positions[i, j] = self.lower_bound + (self.upper_bound - self.lower_bound) * x
        return positions

    def levy_flight(self, L, alpha=1.5):
        sigma1 = np.power((np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2)) /
                          (np.math.gamma((1 + alpha) / 2) * alpha * np.power(2, (alpha - 1) / 2)), 1 / alpha)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=L)
        v = np.random.normal(0, sigma2, size=L)
        step = u / np.power(np.abs(v), 1 / alpha)
        return step

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.positions[i])
                self.evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

                if self.evaluations >= self.budget:
                    break

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                if np.random.rand() < 0.2:  # increased probability for Levy flights
                    self.positions[i] += self.levy_flight(self.dim)
                else:
                    self.positions[i] += self.velocities[i]

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score