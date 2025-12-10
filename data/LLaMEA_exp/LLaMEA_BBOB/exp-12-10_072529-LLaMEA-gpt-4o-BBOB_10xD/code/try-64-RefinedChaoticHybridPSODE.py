import numpy as np

class RefinedChaoticHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 + dim
        self.population_size = self.initial_population_size
        self.c1 = 1.7  # increased cognitive component
        self.c2 = 1.7  # increased social component
        self.w = 0.7   # increased inertia weight for initial exploration
        self.de_f = 0.7  # increased DE scaling factor for better exploration
        self.de_cr = 0.9  # DE crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.velocity_clamp = 0.5 * (self.upper_bound - self.lower_bound)
        self.chaotic_seq = np.random.rand(self.budget) * 0.1  # adjusted chaotic sequence range

    def __call__(self, func):
        evals = 0
        adapt_rate = 0.1
        adapt_lr = 0.2
        cooling_factor = 0.99
        exploration_phase = True
        
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

            # Adjust adaptive parameters dynamically
            chaotic_factor = self.chaotic_seq[evals % self.budget]
            self.w = max(0.1, self.w * cooling_factor + chaotic_factor)
            self.c1 = self.c1 * (1 + adapt_rate * chaotic_factor)
            self.c2 = self.c2 * (1 + adapt_rate * chaotic_factor)

            # Elite retention strategy: preserve the best solutions
            elites = np.argsort(self.pbest_scores)[:max(2, self.population_size // 8)]
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest_position - self.population[i])
                self.velocities[i] = self.w * (self.velocities[i]) + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                if i in elites:
                    continue  # Skip DE mutation for elite solutions

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

            # Stochastic restart if stuck
            if evals % (self.budget // 5) == 0:
                if np.ptp(self.pbest_scores) < 1e-5:
                    self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                    self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
                    self.pbest_positions = np.copy(self.population)
                    self.pbest_scores.fill(np.inf)

        return self.gbest_position, self.gbest_score