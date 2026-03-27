import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_min = 0.4
        self.F_max = 0.9
        self.CR = 0.9  # Crossover probability
        self.init_population()
        self.dynamic_resizing_threshold = 0.1
        self.min_population_size = 4 * dim

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)

    def adapt_mutation_factor(self, score):
        return self.F_min + (self.F_max - self.F_min) * np.exp(-score)

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = self.adapt_mutation_factor(self.scores[target_idx])
        donor_vector = (
            self.population[a] + F * (self.population[b] - self.population[c])
        )
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor):
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < self.CR
        trial[crossover_points] = donor[crossover_points]
        return trial

    def dynamic_population_resizing(self):
        if np.random.rand() < self.dynamic_resizing_threshold and self.population_size > self.min_population_size:
            self.population_size = max(self.min_population_size, int(self.population_size * 0.9))
            self.population = self.population[:self.population_size]
            self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        bounds = func.bounds
        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        while self.budget > 0:
            for i in range(self.population_size):
                donor = self.mutate(i)
                trial = self.crossover(self.population[i], donor)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if self.budget <= 0:
                    break

            self.dynamic_population_resizing()

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution