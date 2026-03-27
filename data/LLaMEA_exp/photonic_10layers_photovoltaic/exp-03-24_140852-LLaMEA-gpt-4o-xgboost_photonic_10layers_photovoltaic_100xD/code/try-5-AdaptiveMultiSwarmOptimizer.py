import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Increased swarm size for diversity
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w = 0.9  # Initial inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 2.0  # Cognitive weight increased for exploration
        self.c2 = 2.0  # Social weight increased for exploration

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.population_size):
                current_score = func(self.particles[i])
                eval_count += 1
                if eval_count >= self.budget:
                    break

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.particles[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.particles[i]

            self.w = self.w_min + (0.9 - self.w_min) * (self.budget - eval_count) / self.budget

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                
                # Introduce Lévy flight for enhanced exploration
                beta = 1.5
                sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                         (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                u = np.random.normal(0, sigma, self.dim)
                v = np.random.normal(0, 1, self.dim)
                step = u / abs(v) ** (1 / beta)
                lévy_flight = 0.1 * step

                self.particles[i] += self.velocities[i] + lévy_flight
                self.particles[i] = np.clip(self.particles[i], lb, ub)

        return self.global_best_position, self.global_best_score