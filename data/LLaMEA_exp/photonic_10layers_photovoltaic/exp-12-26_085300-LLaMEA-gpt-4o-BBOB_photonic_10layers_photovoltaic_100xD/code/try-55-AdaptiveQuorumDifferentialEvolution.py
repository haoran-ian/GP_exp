import numpy as np

class AdaptiveQuorumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.quorum_threshold = 0.2
        self.population = None

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, idx, population, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + self.mutation_factor * (b - c), bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def stochastic_adapt_parameters(self):
        # Stochastic adjustment of mutation factor and crossover probability
        self.mutation_factor = np.random.uniform(0.5, 1.0)
        self.crossover_probability = np.random.uniform(0.7, 1.0)

    def quorum_sensing(self, population, fitness):
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        for i in range(self.population_size):
            if fitness[i] > fitness[best_idx] * (1 + self.quorum_threshold):
                population[i] = best_solution + np.random.normal(0, 0.1, self.dim)

    def elitist_selection(self, population, fitness, new_population, new_fitness):
        for i in range(self.population_size):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]

    def __call__(self, func):
        self.population = self.initialize_population(func.bounds)
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.population_size

        while evaluations < self.budget:
            self.stochastic_adapt_parameters()
            new_population = np.copy(self.population)
            new_fitness = np.copy(fitness)

            for i in range(self.population_size):
                mutant_vector = self.mutate(i, self.population, func.bounds)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < new_fitness[i]:
                    new_population[i] = trial_vector
                    new_fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

            self.elitist_selection(self.population, fitness, new_population, new_fitness)
            self.quorum_sensing(self.population, fitness)

        best_idx = np.argmin(fitness)
        return self.population[best_idx]