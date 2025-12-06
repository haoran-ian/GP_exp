import numpy as np

class AdaptiveLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

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

            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))
            c1 = self.c1_initial - (1.5 * (self.evaluations / self.budget))
            c2 = self.c2_initial + (1.5 * (self.evaluations / self.budget))

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity

                if np.random.rand() < 0.15:
                    self.positions[i] += self.levy_flight(self.dim)
                else:
                    self.positions[i] += self.velocities[i]

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score