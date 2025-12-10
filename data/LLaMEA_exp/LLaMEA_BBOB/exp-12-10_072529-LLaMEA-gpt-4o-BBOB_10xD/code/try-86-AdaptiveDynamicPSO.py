import numpy as np

class AdaptiveDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 + dim
        self.population_size = self.initial_population_size
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.velocity_clamp = 0.5 * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        evals = 0
        improvement_threshold = 1e-5
        no_improvement_count = 0
        
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
                    no_improvement_count = 0

            if evals >= self.budget:
                break

            # Dynamic adjustment of learning rates and inertia
            improvement = np.abs(np.min(self.pbest_scores) - self.gbest_score)
            if improvement < improvement_threshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count > self.budget // 10:  # If no improvement over a significant period
                self.population_size = min(self.population_size + 5, self.initial_population_size * 2)
                self.population = np.vstack([self.population, np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))])
                self.velocities = np.vstack([self.velocities, np.random.uniform(-1, 1, (5, self.dim))])
                self.pbest_positions = np.vstack([self.pbest_positions, np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))])
                self.pbest_scores = np.concatenate([self.pbest_scores, np.full(5, np.inf)])

            self.w = 0.5 + 0.4 * (1 - evals / self.budget)
            self.c1 = 1.5 + 0.5 * (1 - evals / self.budget)
            self.c2 = 2.5 - 0.5 * (1 - evals / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.gbest_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            if np.random.rand() < 0.01:  # Stochastic restarts
                self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
                self.pbest_positions = np.copy(self.population)
                self.pbest_scores.fill(np.inf)

        return self.gbest_position, self.gbest_score