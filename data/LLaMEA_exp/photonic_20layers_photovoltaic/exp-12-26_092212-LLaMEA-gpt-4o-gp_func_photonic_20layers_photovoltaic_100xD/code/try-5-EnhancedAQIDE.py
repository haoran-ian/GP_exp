import numpy as np

class EnhancedAQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.CR = 0.9
        self.F = 0.5
        self.population = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.best_solution = self.population[0]

    def mutate(self, idx):
        sorted_indices = np.argsort([func(ind) for ind in self.population])
        indices = [i for i in range(self.population_size) if i != idx]
        weights = np.linspace(0, 1, len(indices))
        a, b, c = np.random.choice(indices, 3, replace=False, p=weights)
        adaptive_F = self.F * (1 - (self.best_score * self.evaluations) / (self.budget + 1e-9))
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, iteration):
        adaptive_CR = self.CR * (1 - iteration / self.budget)
        j_rand = np.random.randint(self.dim)
        trial = np.array([mutant[j] if np.random.rand() < adaptive_CR or j == j_rand else target[j] 
                          for j in range(self.dim)])
        return trial

    def adaptive_population_resize(self):
        self.population_size = max(5, int(self.initial_population_size * (1 - self.evaluations / self.budget)))
        self.population = self.population[:self.population_size]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant, self.evaluations)
                
                if self.evaluations >= self.budget:
                    break

                score = func(trial)
                self.evaluations += 1

                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = trial

                if score < func(self.population[i]):
                    self.population[i] = trial

            if self.evaluations < self.budget:
                self.adaptive_population_resize()

        return self.best_solution