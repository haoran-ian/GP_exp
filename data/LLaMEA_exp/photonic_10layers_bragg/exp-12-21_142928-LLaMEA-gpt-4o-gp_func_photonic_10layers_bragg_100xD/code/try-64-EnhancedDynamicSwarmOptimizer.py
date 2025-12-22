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

    def adaptive_mutation_strength(self, eval_count):
        return 0.5 + 0.5 * np.cos(np.pi * eval_count / self.budget)

    def stochastic_ranking(self, scores):
        ranks = np.argsort(scores)
        return ranks

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        eval_count = 0
        while eval_count < self.budget:
            scores = np.array([func(self.particles[i]) for i in range(self.population_size)])
            ranks = self.stochastic_ranking(scores)

            for rank in ranks:
                score = scores[rank]

                if score < self.personal_best_scores[rank]:
                    self.personal_best_scores[rank] = score
                    self.personal_best[rank] = self.particles[rank].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[rank].copy()

            inertia_weight = 0.9 - (0.8 * eval_count / self.budget) + 0.1 * np.cos(np.pi * eval_count / self.budget)
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            self.velocities = (
                inertia_weight * self.velocities
                + cognitive_component * (self.personal_best - self.particles)
                + social_component * (self.global_best - self.particles)
            )
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            adaptive_strength = self.adaptive_mutation_strength(eval_count)
            
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                self.f = adaptive_strength * (0.5 + 0.3 * (eval_count / self.budget) * np.random.rand())
                mutant_vector = self.personal_best[a] + self.f * (self.personal_best[b] - self.personal_best[c])
                mutant_vector = np.clip(mutant_vector, lower_bound, upper_bound)

                trial_vector = np.where(
                    np.random.rand(self.dim) < self.cr,
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

        return self.global_best