import numpy as np

class EnhancedAdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 + dim
        self.population_size = self.initial_population_size
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.5   # inertia weight
        self.de_f = 0.5  # DE scaling factor
        self.de_cr = 0.9  # DE crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        evals = 0
        adapt_rate = 0.1
        cooling_factor = 0.98
        mutation_strength = 0.2
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

            fitness_variance = np.var(self.pbest_scores)
            self.w = max(0.1, self.w * (1 - adapt_rate * (self.gbest_score / np.mean(self.pbest_scores))))
            self.de_f = max(0.1, self.de_f * (1 + adapt_rate * (1 - self.gbest_score / (np.mean(self.pbest_scores) + 1e-8))))
            self.c1 = min(2.0, self.c1 + adapt_rate * (np.mean(self.pbest_scores) / (self.gbest_score + 1e-8)))

            self.w *= cooling_factor

            if evals > self.budget * 0.3:
                self.de_cr = min(1.0, self.de_cr + 0.05 * (1 - fitness_variance))

            mutation_strength = max(0.05, mutation_strength * cooling_factor * 0.98)

            if exploration_phase and evals > self.budget * 0.5:
                exploration_phase = False
                self.population_size = max(5, self.population_size // 2)
                adapt_rate += 0.05

            elites = np.argsort(self.pbest_scores)[:max(2, self.population_size // 10)]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest_position - self.population[i])
                noise = np.random.normal(0, mutation_strength, self.dim)
                self.velocities[i] = self.w * (self.velocities[i] + noise) + cognitive + social
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

        return self.gbest_position, self.gbest_score