import numpy as np

class AdaptiveLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.initial_w = 0.9  # dynamic inertia starts high
        self.final_w = 0.4  # dynamic inertia ends low
        self.c1 = 2.05  # cognitive component
        self.c2 = 2.0  # social component
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

    def adaptive_mutation(self, diversity):
        return np.random.normal(0, diversity, self.dim)

    def update_inertia_weight(self):
        return self.final_w + (self.initial_w - self.final_w) * ((self.budget - self.evaluations) / self.budget)

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.zeros(self.num_particles)  # to store particle scores
            for i in range(self.num_particles):
                scores[i] = func(self.positions[i])
                self.evaluations += 1
                
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

                if self.evaluations >= self.budget:
                    break

            diversity = np.std(scores) + 1e-8  # ensure no division by zero
            self.w = self.update_inertia_weight()

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])

                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                if np.random.rand() < 0.1:  # reduced probability of levy flight
                    self.positions[i] += self.levy_flight(self.dim)
                elif np.random.rand() < 0.05:  # additional mutation based on diversity
                    self.positions[i] += self.adaptive_mutation(diversity)
                else:
                    self.positions[i] += self.velocities[i]

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score