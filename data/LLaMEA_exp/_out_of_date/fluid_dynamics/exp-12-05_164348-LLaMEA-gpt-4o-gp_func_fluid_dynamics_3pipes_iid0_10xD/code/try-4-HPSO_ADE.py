import numpy as np

class HPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.inertia = 0.9  # Adjusted inertia for improved exploration
        self.cognitive = 1.5
        self.social = 1.5
        self.mutation_factor = 0.1  # New mutation factor for dynamic adaptation

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the current population
            scores = np.array([func(ind) for ind in self.population])

            # Update personal bests
            better_indices = scores < self.pbest_scores
            self.pbest_positions[better_indices] = self.population[better_indices]
            self.pbest_scores[better_indices] = scores[better_indices]

            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.gbest_score:
                self.gbest_score = scores[min_score_idx]
                self.gbest_position = self.population[min_score_idx]

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            self.velocities = (self.inertia * self.velocities +
                               self.cognitive * r1 * (self.pbest_positions - self.population) +
                               self.social * r2 * (self.gbest_position - self.population))
            self.population += self.velocities
            self.population = np.clip(self.population, -5, 5)

            # Apply Differential Evolution (DE) crossover and mutation
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.f * (b - c), -5, 5)
                trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, self.population[i]) + self.mutation_factor * np.random.normal(size=self.dim)
                trial_score = func(trial_vector)
                if trial_score < self.pbest_scores[i]:
                    self.population[i] = trial_vector
                    self.pbest_scores[i] = trial_score
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest_position = trial_vector

        return self.gbest_position, self.gbest_score