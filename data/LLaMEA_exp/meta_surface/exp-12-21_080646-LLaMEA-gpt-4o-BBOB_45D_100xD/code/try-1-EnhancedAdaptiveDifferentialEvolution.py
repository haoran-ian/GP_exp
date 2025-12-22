import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.population = None
        self.bounds = None
        self.fitness = None
        self.crossover_rate = 0.5
        self.mutation_factor = 0.8
        self.generations = 0

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_fitness(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def mutate(self, target_idx):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.adaptive_mutation_factor() * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def adaptive_mutation_factor(self):
        return self.mutation_factor + np.random.rand() * 0.2

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial_vector, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness

    def adapt_parameters(self):
        self.crossover_rate = np.random.rand() * 0.5 + 0.25
        self.mutation_factor = np.random.rand() * 0.4 + 0.6
        if self.generations % 10 == 0:
            self.population_size = max(4, int(0.9 * self.population_size))
            self.reinitialize_population()

    def reinitialize_population(self):
        lb, ub = self.bounds.lb, self.bounds.ub
        self.population = np.vstack((self.population[:self.population_size],
                                     lb + (ub - lb) * np.random.rand(self.initial_population_size - self.population_size, self.dim)))
        self.fitness = np.hstack((self.fitness[:self.population_size],
                                  np.full(self.initial_population_size - self.population_size, np.inf)))

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)
        self.evaluate_fitness(func)

        evaluations = self.population_size
        while evaluations < self.budget:
            self.adapt_parameters()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)
                if evaluations >= self.budget:
                    break
            self.generations += 1

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]