import numpy as np

class AQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Scaling population size with dimensionality
        self.CR = 0.85  # Adjusted crossover probability
        self.F = 0.5  # Mutation factor
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
        return np.random.normal(mean, std, self.dim)

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        # Modify F based on iteration and best score
        adaptive_F = self.F * (1 - (self.best_score * self.evaluations) / (self.budget + 1e-9))
        mutant = self.population[a] + adaptive_F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.bounds.lb, self.bounds.ub)

    def crossover(self, target, mutant, iteration):
        adaptive_CR = self.CR * (1 - iteration / self.budget)
        j_rand = np.random.randint(self.dim)
        trial = np.array([mutant[j] if np.random.rand() < adaptive_CR or j == j_rand else target[j] 
                          for j in range(self.dim)])
        return trial

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
                quantum_candidate = self.quantum_superposition()
                q_score = func(quantum_candidate)
                self.evaluations += 1

                if q_score < self.best_score:
                    self.best_score = q_score
                    self.best_solution = quantum_candidate

        return self.best_solution