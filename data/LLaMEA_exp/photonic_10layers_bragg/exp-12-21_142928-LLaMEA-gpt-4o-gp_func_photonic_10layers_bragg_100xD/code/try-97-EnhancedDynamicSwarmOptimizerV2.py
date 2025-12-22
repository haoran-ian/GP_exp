import numpy as np

class EnhancedDynamicSwarmOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')
        self.mutation_rate = 0.9

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def adaptive_chaotic_map(self, x, it):
        a = 0.9 + 0.1 * (it / self.budget)
        return a * 4 * x * (1 - x)

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            inertia_weight = (0.9 - 0.4 * (eval_count / self.budget)) + 0.1 * np.cos(np.pi * eval_count / self.budget)
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            self.velocities = (
                inertia_weight * self.velocities
                + cognitive_component * (self.personal_best - self.particles)
                + social_component * (self.global_best - self.particles)
            )
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                f = 0.5 + 0.3 * (eval_count / self.budget) * np.random.rand()
                mutant_vector = self.personal_best[a] + f * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = self.adaptive_chaotic_map(mutant_vector[i % self.dim], eval_count) * (upper_bound - lower_bound) + lower_bound
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

                trial_vector = np.where(
                    np.random.rand(self.dim) < self.mutation_rate,
                    mutant_vector,
                    self.particles[i]
                )
                
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

            if (eval_count / self.budget) > 0.5:
                self.population_size = max(10, int(self.initial_population_size * (1 - (eval_count / self.budget))))
                self.mutation_rate *= 0.95  # Gradually decrease mutation rate for finer exploitation

        return self.global_best