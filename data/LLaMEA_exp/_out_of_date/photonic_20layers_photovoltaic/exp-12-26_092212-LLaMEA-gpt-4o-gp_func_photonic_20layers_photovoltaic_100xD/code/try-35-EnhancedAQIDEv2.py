import numpy as np

class EnhancedAQIDEv2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, int(10 * np.log(dim + 1)))
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

    def quantum_superposition(self):
        mean = np.mean(self.population, axis=0)
        std = np.std(self.population, axis=0) / np.sqrt(self.population_size)
        return np.random.normal(mean, std * 0.8, self.dim)

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_F = self.F * (1 - (self.best_score * np.exp(-self.evaluations / self.budget)))
        diversity_factor = 0.5 + (0.5 * (1 - self.evaluations / self.budget))
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c]) * diversity_factor
        noise = np.random.normal(0, 0.1 * (1 - self.evaluations / self.budget), self.dim)
        return np.clip(mutant + noise, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, iteration):
        adaptive_CR = self.CR * (1 - (iteration / self.budget)**0.5)
        j_rand = np.random.randint(self.dim)
        trial = np.empty(self.dim)
        for j in range(self.dim):
            if np.random.rand() < adaptive_CR or j == j_rand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j] if np.random.rand() > 0.5 else self.best_solution[j]
        return trial

    def dynamic_population_adjustment(self):
        if self.evaluations > self.budget * 0.5:
            shrink_factor = 0.6
            self.population_size = max(5, int(self.population_size * shrink_factor))
            self.population = self.population[:self.population_size]

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds)

        while self.evaluations < self.budget:
            self.dynamic_population_adjustment()
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
                quantum_candidate = self.quantum_superposition() * 1.0
                q_score = func(quantum_candidate)
                self.evaluations += 1

                if q_score < self.best_score:
                    self.best_score = q_score
                    self.best_solution = quantum_candidate

        return self.best_solution