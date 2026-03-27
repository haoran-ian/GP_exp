import numpy as np

class AdvancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, self.budget // 10)
        self.population_size = self.initial_population_size
        self.f = 0.5  # Initial differential weight
        self.cr = 0.9  # Initial crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.best_fitness = np.inf
        self.lambda_scale = 1.0

    def chaotic_map(self, value):
        return 4.0 * value * (1.0 - value)

    def adapt_parameters(self, generation):
        self.f = max(0.1, self.chaotic_map(self.f))
        self.cr = min(1.0, self.chaotic_map(self.cr))
        self.lambda_scale *= 0.99  # Gradually reduce the scale to focus the search

    def adjust_rates(self, trial_fitness, current_fitness):
        if trial_fitness < current_fitness:
            self.f = min(0.9, self.f + 0.05)
            self.cr = max(0.1, self.cr - 0.05)
            self.lambda_scale = min(1.0, self.lambda_scale * 1.01)  # Slightly increase scale if improving

    def adapt_population_size(self):
        if self.budget - self.evaluations < self.population_size:
            self.population_size = max(5, (self.budget - self.evaluations) // 2)

    def __call__(self, func):
        generation = 0
        while self.evaluations < self.budget:
            self.adapt_population_size()
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.adapt_parameters(generation)

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = self.f * self.lambda_scale
                mutant = self.population[a] + mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, self.population[i])

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.adjust_rates(trial_fitness, self.fitness[i])

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness

            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            self.population[0] = elite

            generation += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]