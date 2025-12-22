import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')
        self.f = 0.8  # Initial differential weight
        self.cr = 0.9  # Crossover probability

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def adaptive_chaotic_map(self, x, it):
        # Introducing an adaptive chaotic map
        a = 0.9 + 0.1 * (it / self.budget)
        return a * 4 * x * (1 - x)

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        eval_count = 0
        while eval_count < self.budget:
            # Evaluate particles
            for i in range(self.population_size):
                score = func(self.particles[i])

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update velocities and positions for PSO
            inertia_weight = 0.85 - (0.8 * (eval_count / self.budget)**2)  # Adjusted non-linear inertia weight
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            dynamic_scaling_factor = 0.5 + 0.5 * (eval_count / self.budget)  # New dynamic scaling factor
            self.velocities = (
                dynamic_scaling_factor * inertia_weight * self.velocities
                + cognitive_component * (self.personal_best - self.particles)
                + social_component * (self.global_best - self.particles)
            )
            self.particles += self.velocities

            # Apply boundaries
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            # Differential Evolution mutation and crossover with diversity boost
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                self.f = 0.5 + 0.3 * (eval_count / self.budget) * np.random.rand()  # Dynamic differential weight
                mutant_vector = self.personal_best[a] + self.f * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = self.adaptive_chaotic_map(mutant_vector[i % self.dim], eval_count) * (upper_bound - lower_bound) + lower_bound
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

                trial_vector = np.where(
                    np.random.rand(self.dim) < self.cr,
                    mutant_vector, 
                    self.particles[i]
                )

                # Integrating dynamic Lévy flights with diversity boost
                levy_factor = 0.1 * (1 - (eval_count / self.budget))
                trial_vector += levy_factor * self.levy_flight(self.dim)

                trial_vector = np.clip(trial_vector, lower_bound, upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector.copy()
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector.copy()

        return self.global_best