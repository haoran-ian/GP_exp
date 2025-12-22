import numpy as np

class AdaptiveDiversityGuidedDifferentialEvolution:
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
        self.dynamic_population_control = True
        self.diversity_threshold = 0.1  # New: threshold to monitor diversity

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
        mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.bounds.lb, self.bounds.ub)
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target_idx, trial_vector, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness

    def adapt_parameters(self):
        # New: Adapt parameters based on diversity
        diversity = np.mean(np.std(self.population, axis=0))
        if diversity < self.diversity_threshold:
            self.crossover_rate = 0.1 + np.random.rand() * 0.9
            self.mutation_factor = 0.9  # Increase mutation factor if diversity is low
        else:
            self.crossover_rate = 0.1 + np.random.rand() * 0.9
            self.mutation_factor = 0.6 + np.random.rand() * 0.4

    def control_population_size(self):
        best_fitness = np.min(self.fitness)
        if self.dynamic_population_control and best_fitness < 1e-5:
            self.population_size = max(int(self.population_size * 0.9), self.dim * 2)

    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population(self.bounds.lb, self.bounds.ub)
        self.evaluate_fitness(func)

        evaluations = self.population_size
        while evaluations < self.budget:
            self.adapt_parameters()
            self.control_population_size()
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                self.select(i, trial, trial_fitness)
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]