import numpy as np

class HPSO_ADE_Refined:
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
        self.f = 0.5
        self.cr = 0.9
        self.inertia = 0.7
        self.cognitive = 1.7
        self.social = 1.5
        self.chaos = np.random.rand()  # Introduce chaotic variable

    def __call__(self, func):
        for iteration in range(self.budget):
            self.inertia = 0.9 - 0.6 * (iteration / self.budget)  # Change 1: Slightly refined adaptive inertia
            self.social = 1.6 + 0.4 * (iteration / self.budget)    # Change 2: Slightly refined adaptive social factor
            self.f = 0.6 + 0.2 * np.sin(0.6 * np.pi * iteration / self.budget)  # Change 3
            self.cr = 0.85 - 0.35 * (iteration / self.budget)  # Change 4

            scores = np.array([func(ind) for ind in self.population])
            better_indices = scores < self.pbest_scores
            self.pbest_positions[better_indices] = self.population[better_indices]
            self.pbest_scores[better_indices] = scores[better_indices]

            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.gbest_score:
                self.gbest_score = scores[min_score_idx]
                self.gbest_position = self.population[min_score_idx]

            elite_size = max(1, int(5 * (1 - iteration / self.budget)))
            elite_indices = np.argsort(scores)[:elite_size]
            elite_population = self.population[elite_indices]

            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            self.velocities = (self.inertia * self.velocities +
                               self.cognitive * r1 * (self.pbest_positions - self.population) +
                               self.social * r2 * (self.gbest_position - self.population))
            self.population += self.velocities
            self.population = np.clip(self.population, -5, 5)

            for i in range(self.population_size):
                if i not in elite_indices:
                    indices = [index for index in range(self.population_size) if index != i]
                    a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                    mutant_vector = np.clip(a + self.f * (b - c), -5, 5)
                    trial_vector = np.where(np.random.rand(self.dim) < self.cr, mutant_vector, self.population[i])
                    trial_score = func(trial_vector)
                    if trial_score < self.pbest_scores[i]:
                        self.population[i] = trial_vector
                        self.pbest_scores[i] = trial_score
                        if trial_score < self.gbest_score:
                            self.gbest_score = trial_score
                            self.gbest_position = trial_vector

            self.population[elite_indices] = elite_population
            self.chaos = 4 * self.chaos * (1 - self.chaos)  # Change 5: Chaotic map for enhancing diversity

        return self.gbest_position, self.gbest_score