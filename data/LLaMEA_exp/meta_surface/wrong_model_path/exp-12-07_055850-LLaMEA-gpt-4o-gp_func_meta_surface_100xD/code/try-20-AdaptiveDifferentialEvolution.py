import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.fitness = None

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def select_parents(self):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        return self.population[idxs]

    def mutate(self, target, parents):
        mutant = parents[0] + self.mutation_factor * (parents[1] - parents[2])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate:
                trial[j] = mutant[j]
        return trial

    def select(self, target_idx, trial, func):
        trial_fitness = func(trial)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = trial_fitness

    def adapt_parameters(self, generation):
        population_diversity = np.std(self.population)
        fitness_variance = np.var(self.fitness)
        self.mutation_factor = (0.5 + 0.3 * np.sin(0.5 * generation) * (1 - generation / self.budget)) * (1 + population_diversity * 0.2)  # Slightly increased factor
        self.crossover_rate = 0.7 + 0.2 * np.cos(0.3 * generation) * (1 + fitness_variance * 0.01)

    def __call__(self, func):
        self.initialize_population()
        self.evaluate_population(func)

        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            generation += 1
            self.adapt_parameters(generation)

            for i in range(self.population_size):
                target = self.population[i]
                parents = self.select_parents()
                mutant = self.mutate(target, parents)
                trial = self.crossover(target, mutant)
                self.select(i, trial, func)
                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]