import numpy as np

class Advanced_Adaptive_DE_SGD_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.scaling_factor = 0.8
        self.crossover_rate = 0.9
        self.current_evals = 0
        self.learning_rate = 0.01  # For SGD-inspired updates

        # Initialize population
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_position = np.zeros(self.dim)
        self.best_score = float('inf')
        self.scores = np.full(self.population_size, float('inf'))

    def __call__(self, func):
        while self.current_evals < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.scaling_factor * (b - c)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_score = func(trial)
                self.current_evals += 1
                if trial_score < self.scores[i]:
                    self.scores[i] = trial_score
                    self.population[i] = trial
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = trial

            # Adaptive scaling and crossover rates
            self.scaling_factor = 0.5 + 0.5 * np.sin((np.pi * self.current_evals) / (2.0 * self.budget))
            self.crossover_rate = 0.7 + 0.3 * np.cos((np.pi * self.current_evals) / (2.0 * self.budget))

            # Stochastic Gradient Descent inspired updates
            for i in range(self.population_size):
                gradient = np.random.normal(0, 1, self.dim) * (self.population[i] - self.best_position)
                self.population[i] -= self.learning_rate * gradient
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        return self.best_position, self.best_score