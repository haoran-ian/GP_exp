import numpy as np

class AdaptiveLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.w = 0.9  # adaptive inertia weight, start with a higher value
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 2.0  # slightly reduced cognitive component
        self.c2 = 2.0  # social component remains unchanged
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

    def differential_mutation(self, target_idx):
        indices = list(range(self.num_particles))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.5  # mutation factor
        mutant_vector = self.positions[a] + F * (self.positions[b] - self.positions[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def adapt_inertia(self):
        self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

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

            self.adapt_inertia()  # adapt inertia weight

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                if np.random.rand() < 0.15:
                    self.positions[i] += self.levy_flight(self.dim)
                elif np.random.rand() < 0.2:  # additional differential mutation
                    self.positions[i] = self.differential_mutation(i)
                else:
                    self.positions[i] += self.velocities[i]

                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score