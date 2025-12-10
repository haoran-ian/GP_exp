import numpy as np

class EnhancedChaoticAdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + dim
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.de_f = 0.6
        self.de_cr = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.velocity_clamp = 0.5 * (self.upper_bound - self.lower_bound)
        self.chaotic_seq = np.random.rand(self.budget) * 0.1
        self.exploration_phase = True

    def __call__(self, func):
        evals = 0
        adaptation_factor = 0.1
        mutation_strength = 0.2
        
        while evals < self.budget:
            for i in range(self.population_size):
                score = func(self.population[i])
                evals += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.population[i]

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.population[i]

            if evals >= self.budget:
                break

            chaotic_factor = self.chaotic_seq[evals % self.budget]
            self.w = max(0.1, self.w * (1 - adaptation_factor * (self.gbest_score / np.mean(self.pbest_scores))) + chaotic_factor)
            self.de_f = max(0.1, self.de_f * (1 + adaptation_factor * (1 - self.gbest_score / (np.mean(self.pbest_scores) + 1e-8))) + chaotic_factor)
            self.c1 = min(2.0, self.c1 + adaptation_factor * (np.mean(self.pbest_scores) / self.gbest_score) + chaotic_factor)

            if evals > self.budget * 0.4:
                self.de_cr = min(1.0, self.de_cr + 0.05)

            if self.exploration_phase and evals > self.budget * 0.6:
                self.exploration_phase = False
                self.population_size = max(5, self.population_size // 2)
                adaptation_factor += 0.05

            elites = np.argsort(self.pbest_scores)[:max(2, self.population_size // 10)]
            adaptive_factor = 0.5 + 0.5 * np.sin((evals / self.budget) * np.pi)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest_position - self.population[i])
                noise = np.random.normal(0, mutation_strength, self.dim)
                self.velocities[i] = adaptive_factor * self.w * (self.velocities[i] + noise) + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                if i in elites:
                    continue

                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant_vector = np.clip(x1 + self.de_f * (x2 - x3 + (self.gbest_position - x1)), self.lower_bound, self.upper_bound)

                trial_vector = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.de_cr
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                trial_score = func(trial_vector)
                evals += 1

                if trial_score < func(self.population[i]):
                    self.population[i] = trial_vector

            if evals % (self.budget // 3) == 0:
                if np.ptp(self.pbest_scores) < 1e-6:
                    self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
                    self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
                    self.pbest_positions = np.copy(self.population)
                    self.pbest_scores.fill(np.inf)

        return self.gbest_position, self.gbest_score