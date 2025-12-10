import numpy as np

class ImprovedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(50, self.budget // 10)
        self.population_size = self.initial_population_size
        self.f = 0.5
        self.cr = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.best_fitness = np.inf

    def chaotic_map(self, value):
        return 4.0 * value * (1.0 - value)

    def diversity(self):
        return np.mean(np.std(self.population, axis=0))

    def adapt_parameters(self, generation, diversity_level):
        self.f = max(0.1, self.chaotic_map(self.f))
        self.cr = min(1.0, self.chaotic_map(self.cr))
        if diversity_level < 0.1:
            self.f = max(0.1, self.f * 0.9)
            self.cr = min(1.0, self.cr * 1.1)
    
    def adjust_rates(self, trial_fitness, current_fitness):
        if trial_fitness < current_fitness:
            self.f = min(0.9, self.f + 0.05)
            self.cr = max(0.1, self.cr - 0.05)

    def adapt_population_size(self):
        if self.budget - self.evaluations < self.population_size:
            self.population_size = max(5, (self.budget - self.evaluations) // 2)

    def __call__(self, func):
        generation = 0
        while self.evaluations < self.budget:
            self.adapt_population_size()
            self.population = self.population[:self.population_size]
            self.fitness = self.fitness[:self.population_size]

            diversity_level = self.diversity()

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.adapt_parameters(generation, diversity_level)

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
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