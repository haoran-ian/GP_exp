import numpy as np

class EnhancedChaoticMPAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 + dim
        self.population_size = self.initial_population_size
        self.c1 = 2.0  # Increased cognitive component
        self.c2 = 2.0  # Increased social component
        self.w = 0.7   # Increased inertia weight
        self.de_f = 0.7  # Increased DE scaling factor
        self.de_cr = 0.8  # Adjusted DE crossover probability
        self.populations = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim)) for _ in range(3)]
        self.velocities = [np.random.uniform(-1, 1, (self.population_size, dim)) for _ in range(3)]
        self.pbest_positions = [np.copy(pop) for pop in self.populations]
        self.pbest_scores = [np.full(self.population_size, np.inf) for _ in range(3)]
        self.gbest_positions = [None, None, None]
        self.gbest_scores = [np.inf, np.inf, np.inf]
        self.velocity_clamp = 0.5 * (self.upper_bound - self.lower_bound)
        self.chaotic_seq = np.random.rand(3, self.budget) * 0.15  # Chaotic sequences per population

    def __call__(self, func):
        evals = 0
        adapt_rate = 0.1
        adapt_lr = 0.2
        mutation_strength = 0.3
        cooling_factor = 0.99  # Faster cooling
        exploration_phase = True
        
        while evals < self.budget:
            for pop_idx in range(3):  # Handle multiple populations
                for i in range(self.population_size):
                    score = func(self.populations[pop_idx][i])
                    evals += 1

                    if score < self.pbest_scores[pop_idx][i]:
                        self.pbest_scores[pop_idx][i] = score
                        self.pbest_positions[pop_idx][i] = self.populations[pop_idx][i]

                    if score < self.gbest_scores[pop_idx]:
                        self.gbest_scores[pop_idx] = score
                        self.gbest_positions[pop_idx] = self.populations[pop_idx][i]

                if evals >= self.budget:
                    break

                # Adjust adaptive parameters using chaotic sequence
                chaotic_factor = self.chaotic_seq[pop_idx, evals % self.budget]
                avg_pbest_score = np.mean(self.pbest_scores[pop_idx])
                self.w = max(0.1, self.w * (1 - adapt_lr * adapt_rate * (self.gbest_scores[pop_idx] / avg_pbest_score)) + chaotic_factor)
                self.de_f = max(0.1, self.de_f * (1 + adapt_lr * adapt_rate * (1 - self.gbest_scores[pop_idx] / (avg_pbest_score + 1e-8))) + chaotic_factor)
                self.c1 = min(3.0, self.c1 + adapt_lr * adapt_rate * (avg_pbest_score / self.gbest_scores[pop_idx]) + chaotic_factor)

                # Nonlinear adaptive cooling strategy
                self.w *= cooling_factor ** (evals / self.budget)

                if exploration_phase and evals > self.budget * 0.5:
                    exploration_phase = False
                    self.population_size = max(5, self.population_size // 2)
                    adapt_lr += 0.05

                # Elite retention strategy: preserve the best solutions
                elites = np.argsort(self.pbest_scores[pop_idx])[:max(2, self.population_size // 10)]
                
                # Quasi-sinusoidal adaptive strategy for velocity
                angle = (evals / self.budget) * np.pi
                adaptive_factor = 0.5 + 0.5 * np.sin(angle)

                for i in range(self.population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = self.c1 * r1 * (self.pbest_positions[pop_idx][i] - self.populations[pop_idx][i])
                    social = self.c2 * r2 * (self.gbest_positions[pop_idx] - self.populations[pop_idx][i])
                    noise = np.random.normal(0, mutation_strength, self.dim)
                    self.velocities[pop_idx][i] = adaptive_factor * self.w * (self.velocities[pop_idx][i] + noise) + cognitive + social
                    self.velocities[pop_idx][i] = np.clip(self.velocities[pop_idx][i], -self.velocity_clamp, self.velocity_clamp)
                    self.populations[pop_idx][i] = np.clip(self.populations[pop_idx][i] + self.velocities[pop_idx][i], self.lower_bound, self.upper_bound)

                    if i in elites:
                        continue  # Skip DE mutation for elite solutions

                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x1, x2, x3 = self.populations[pop_idx][indices]
                    mutant_vector = np.clip(x1 + self.de_f * (x2 - x3 + (self.gbest_positions[pop_idx] - x1)), self.lower_bound, self.upper_bound)

                    trial_vector = np.copy(self.populations[pop_idx][i])
                    crossover_mask = np.random.rand(self.dim) < self.de_cr
                    trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                    trial_score = func(trial_vector)
                    evals += 1

                    if trial_score < func(self.populations[pop_idx][i]):
                        self.populations[pop_idx][i] = trial_vector

                # Stochastic restart if stuck
                if evals % (self.budget // 4) == 0:
                    if np.ptp(self.pbest_scores[pop_idx]) < 1e-6:
                        self.populations[pop_idx] = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
                        self.velocities[pop_idx] = np.random.uniform(-1, 1, (self.population_size, dim))
                        self.pbest_positions[pop_idx] = np.copy(self.populations[pop_idx])
                        self.pbest_scores[pop_idx].fill(np.inf)

        best_global_idx = np.argmin(self.gbest_scores)
        return self.gbest_positions[best_global_idx], self.gbest_scores[best_global_idx]